# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MNIST model training using TPUs.

This program demonstrates training of the convolutional neural network model
defined in mnist.py on Google Cloud TPUs (https://cloud.google.com/tpu/).

If you are not interested in TPUs, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import time
from time import localtime, strftime

from official.mnist import dataset
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


LEARNING_RATE = 1e-4

# For open source environment, add grandparent directory for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("eval_batch_size", 1000,
                        "Mini-batch size for the eval. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")

tf.flags.DEFINE_integer("warm_up_epochs", 0, "Total number of epochs for warming up.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.flags.DEFINE_float("poly_power", 0.5, "The power of poly decay scheme.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS

def create_model(data_format):
  """Model to recognize digits in the MNIST dataset.
  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
  But uses the tf.keras API.
  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])


def metric_fn(labels, logits):
  accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.argmax(logits, axis=1))
  return {"accuracy": accuracy}


def model_fn(features, labels, mode, params):
  """model_fn constructs the ML model used to predict handwritten digits."""
  del params
  image = features
  if isinstance(image, dict):
    image = features["image"]

  model = create_model("channels_last")

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'class_ids': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

  logits = model(image, training=(mode == tf.estimator.ModeKeys.TRAIN))
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  warm = tf.Variable(0.0, dtype=tf.float32)

  if mode == tf.estimator.ModeKeys.TRAIN:
    warm = 60000.0*FLAGS.warm_up_epochs/tf.cast(FLAGS.batch_size, tf.float32)
    g_step = tf.cast(tf.train.get_global_step(), tf.float32)
    learning_rate = tf.cond(tf.greater(warm, g_step), lambda : (g_step/warm*FLAGS.learning_rate), lambda : tf.train.polynomial_decay(FLAGS.learning_rate, g_step, FLAGS.train_steps, end_learning_rate=0.0001, power=FLAGS.poly_power, cycle=False, name=None))
    # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, tf.train.get_global_step(), 100000, 0.95, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    print("++++++++++++++++++++++++ I'm using Momentum Optimizer ++++++++++++++++++++++++")
    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_global_step()))

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


def train_input_fn(params):
  """train_input_fn defines the input pipeline used for training."""
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  # Retrieves the batch size for the current shard. The # of shards is
  # computed according to the input pipeline deployment. See
  # `tf.contrib.tpu.RunConfig` for details.
  ds = dataset.train(data_dir).cache().repeat().shuffle(
      buffer_size=60000).batch(batch_size, drop_remainder=True)
  return ds


def eval_input_fn(params):
  #batch_size = params["batch_size"]
  batch_size = 1000
  data_dir = params["data_dir"]
  ds = dataset.test(data_dir).batch(batch_size, drop_remainder=True)
  return ds


def predict_input_fn(params):
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  # Take out top 10 samples from test data to make the predictions.
  ds = dataset.test(data_dir).take(10).batch(batch_size)
  return ds


def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)
  file = open("batch-"+str(FLAGS.batch_size)+"-lr-"+str(FLAGS.learning_rate)+"-warmup-"+
  str(FLAGS.warm_up_epochs)+"-poly-"+str(FLAGS.poly_power)+"-"+strftime("%Y-%m-%d-%H-%M-%S", localtime())+".log",'w')
  file.write("****** MNIST by with Poly Decay ******\n")
  file.write("batch_size: " + str(FLAGS.batch_size) + "; ")
  file.write("init_lr: " + str(FLAGS.learning_rate) + "; ")
  file.write("poly_power: " + str(FLAGS.poly_power) + "; ")
  file.write("warm_up: " + str(FLAGS.warm_up_epochs) + "\n")

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.eval_batch_size,
      params={"data_dir": FLAGS.data_dir},
      config=run_config)
  # TPUEstimator.train *requires* a max_steps argument.
  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
  # TPUEstimator.evaluate *requires* a steps argument.
  # Note that the number of examples used during evaluation is
  # --eval_steps * --batch_size.
  # So if you change --batch_size then change --eval_steps too.
  if FLAGS.eval_steps:
    eval_results = estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)
    print('\nEvaluation results:\n\t%s\n' % eval_results)
    file.write('\nEvaluation results:\n\t%s\n' % eval_results)
    file.close()

  # Run prediction on top few samples of test data.
  if FLAGS.enable_predict:
    predictions = estimator.predict(input_fn=predict_input_fn)

    for pred_dict in predictions:
      template = ('Prediction is "{}" ({:.1f}%).')

      class_id = pred_dict['class_ids']
      probability = pred_dict['probabilities'][class_id]

      print(template.format(class_id, 100 * probability))


if __name__ == "__main__":
  tf.app.run()
