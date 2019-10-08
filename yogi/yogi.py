"""Yogi: An adaptive nonconvex optimizer.

Implementation of Additive Averaging.
m_t+1 = beta1*m_t + (1-beta1)*g_t
v_t+1 = v_t + sign(g_t-v_t)(g_t^2)

Experiments show better performance across NLP and Vision tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import slot_creator

FLAGS = flags.FLAGS


class YogiOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Yogi algorithm.

  See Algorithm 2 of go/yogi-opt.
  """

  def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-3,
               activation='sign', init_steps=1000, per_dim_init=False,
               use_locking=False, name='Yogi'):
    """Construct a new Yogi optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A constant trading off adaptivity and noise.
      activation: Use hard sign or soft tanh to determin sign.
      init_steps: Number of steps to use at beginning to estimate v.
      per_dim_init: Init all v same or per dimension.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Yogi".
    """
    super(YogiOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._activation = activation
    self._init_steps = init_steps
    self._per_dim_init = per_dim_init
    self._var_list = {}

    # Tensor version of constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None
    self._init_steps_t = None

  def _prepare(self):
    """See `tf.train.Optimizer._prepare()`."""
    self._lr_t = ops.convert_to_tensor(self._lr, name='learning_rate')
    self._beta1_t = ops.convert_to_tensor(self._beta1, name='beta1')
    self._beta2_t = ops.convert_to_tensor(self._beta2, name='beta2')
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name='epsilon')
    self._init_steps_t = ops.convert_to_tensor(self._init_steps,
                                               name='init_step')

  def _create_slots(self, var_list):
    """See `tf.train.Optimizer._create_slots()`."""
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    graph = None if context.executing_eagerly() else ops.get_default_graph()
    self._var_list[graph] = var_list
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=self._beta1,
                                   name='beta1_power',
                                   colocate_with=first_var)
    self._create_non_slot_variable(initial_value=self._beta2,
                                   name='beta2_power',
                                   colocate_with=first_var)
    self._create_non_slot_variable(initial_value=0,
                                   name='yogi_steps',
                                   colocate_with=first_var)
    if not self._per_dim_init:
      self._create_non_slot_variable(initial_value=0.0,
                                     name='yogi_cum_sum',
                                     colocate_with=first_var)

    # Create slots for the first and second moments, and maximum second moments.
    for v in var_list:
      self._zeros_slot(v, 'm', self._name)
      self._ones_slot(v, 'v', self._name)

  def _get_beta_accumulators(self):
    """Returns beta accumulators.

    Returns:
      Two slot variables, `beta1_power` and `beta2_power`, which are cumulative
      decay rates for the 1st and 2nd moments.
    """
    graph = None if context.executing_eagerly() else ops.get_default_graph()
    return (self._get_non_slot_variable('beta1_power', graph=graph),
            self._get_non_slot_variable('beta2_power', graph=graph))

  def _get_yogi_counter(self):
    """Returns yogi step counter.

    Returns:
      One slot variables, `yogi_steps`, which is counting number of steps.
    """
    graph = None if context.executing_eagerly() else ops.get_default_graph()
    return self._get_non_slot_variable('yogi_steps', graph=graph)

  def _get_yogi_sum(self):
    """Returns yogi cummulative sum.

    Returns:
      One slot variables, `yogi_cum_sum`, which is sum of gradient^2.
    """
    graph = None if context.executing_eagerly() else ops.get_default_graph()
    return self._get_non_slot_variable('yogi_cum_sum', graph=graph)

  def _apply_dense(self, grad, var):
    """See `tf.train.Optimizer._apply_dense()`."""

    gs = self._get_yogi_counter()

    def true_func():
      """Default behaviour."""
      beta1_power, beta2_power = self._get_beta_accumulators()
      beta1_power = math_ops.cast(beta1_power, grad.dtype.base_dtype)
      beta2_power = math_ops.cast(beta2_power, grad.dtype.base_dtype)
      lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
      beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
      beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
      epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)

      lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

      # m_t = beta1 * m + (1 - beta1) * g_t
      m = self.get_slot(var, 'm')
      m_t = state_ops.assign(m, m * beta1_t + grad * (1 - beta1_t),
                             use_locking=self._use_locking)

      # v_t = v + sign(g_t-v)(g_t^2)
      v = self.get_slot(var, 'v')
      grad2 = grad * grad
      if self._activation == 'sign':
        sign = math_ops.sign(v - grad2)
      elif self._activation == 'tanh':
        sign = math_ops.tanh(10*(v - grad2))
      else:
        raise NotImplementedError('Activation function should be sign or tanh')
      v_t = state_ops.assign_sub(v, (1-beta2_t) * sign * grad2,
                                 use_locking=self._use_locking)
      v_sqrt = math_ops.sqrt(v_t)

      var_update = state_ops.assign_sub(var,
                                        lr * m_t / (v_sqrt + epsilon_t),
                                        use_locking=self._use_locking)

      # Create an op that groups all the above operations
      return control_flow_ops.group(*[var_update, m_t, v_t])

    def false_func():
      """Init behaviour."""
      init_steps_t = math_ops.cast(self._init_steps_t, grad.dtype.base_dtype)

      # v_t = v + sign(g_t-v)(g_t^2)
      v = self.get_slot(var, 'v')
      grad2 = grad * grad
      if self._per_dim_init:
        v_t = state_ops.assign_add(v, (grad2-1)/init_steps_t,
                                   use_locking=self._use_locking)
      else:
        cs = self._get_yogi_sum()
        contri = math_ops.cast(tf.reduce_sum(grad2), cs.dtype.base_dtype)
        v_t = state_ops.assign_add(cs, contri,
                                   use_locking=self._use_locking)

      return control_flow_ops.group(*[v_t])

    return tf.cond(gs < self._init_steps_t, false_func, true_func)

  def _resource_apply_dense(self, grad, var):
    """See `tf.train.Optimizer._resource_apply_dense()`."""
    return self._apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    """See `tf.train.Optimizer._apply_sparse()`."""
    raise NotImplementedError('Sparse not supported yet')

  def _resource_apply_sparse(self, grad, var, indices):
    """See `tf.train.Optimizer._resource_apply_spase()`."""
    raise NotImplementedError('Resource sparse not supported')

  def _finish(self, update_ops, name_scope):
    """See `tf.train.Optimizer._finish()`."""
    gs = self._get_yogi_counter()

    def true_func():
      """Default behaviour."""
      # Update the power accumulators.
      with ops.control_dependencies(update_ops):
        beta1_power, beta2_power = self._get_beta_accumulators()
        with ops.colocate_with(beta1_power):
          update_beta1 = beta1_power.assign(
              beta1_power * self._beta1_t, use_locking=self._use_locking)
          update_beta2 = beta2_power.assign(
              beta2_power * self._beta2_t, use_locking=self._use_locking)
      return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                    name=name_scope+'1')

    def false_func():
      """Init behaviour."""
      global_step = tf.train.get_or_create_global_step()
      graph = None if context.executing_eagerly() else ops.get_default_graph()
      num_params = 1.0*np.sum([np.prod(var.shape.as_list())
                               for var in self._var_list[graph]])
      # Update the v_t with current estimates
      with ops.control_dependencies(update_ops):
        with ops.colocate_with(gs):
          update_gs = gs.assign_add(1, use_locking=self._use_locking)
          if not self._per_dim_init:
            cs = self._get_yogi_sum()
            contri = cs/(num_params*self._init_steps)
            update_vs = []
            for var in self._var_list[graph]:
              vv = self.get_slot(var, 'v')
              contri_t = math_ops.cast(contri, vv.dtype.base_dtype)
              update_vs.append(vv.assign(tf.ones_like(vv)*contri_t,
                                         use_locking=self._use_locking))
      # Make sure the global step count does not increments for init steps
      if global_step is not None:
        with ops.colocate_with(global_step):
          if isinstance(global_step, resource_variable_ops.ResourceVariable):
            update_ggs = resource_variable_ops.assign_sub_variable_op(
                global_step.handle,
                ops.convert_to_tensor(1, dtype=global_step.dtype))
          else:
            update_ggs = state_ops.assign_sub(global_step, 1)
      return control_flow_ops.group(*update_ops + [update_gs,
                                                   update_ggs] + update_vs,
                                    name=name_scope+'0')

    return tf.cond(gs < self._init_steps_t, false_func, true_func)

  def _ones_slot(self, var, slot_name, op_name):
    """Find or create a slot initialized with 1.0.

    Args:
      var: A `Variable` object.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for the slot.

    Returns:
      A `Variable` object.
    """
    # pylint: disable=protected-access
    named_slots = self._slot_dict(slot_name)
    if optimizer._var_key(var) not in named_slots:
      dtype = var.dtype
      slot_shape = var.get_shape()
      if slot_shape.is_fully_defined():
        initializer = init_ops.constant_initializer(1, dtype)
        result = slot_creator.create_slot_with_initializer(var, initializer, \
                       slot_shape, dtype, op_name, colocate_with_primary=True)
      else:
        if isinstance(var, variables.Variable):
          slot_shape = array_ops.shape(var.initialized_value())
        else:
          slot_shape = array_ops.shape(var)
        val = array_ops.ones(slot_shape, dtype=dtype)
        result = slot_creator.create_slot(var, val, op_name,
                                          colocate_with_primary=True)
      named_slots[optimizer._var_key(var)] = result
    return named_slots[optimizer._var_key(var)]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

if __name__ == '__main__':
  app.run(main)
