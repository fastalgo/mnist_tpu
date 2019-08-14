export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
export TPU_NAME=v3-8-tpu
export DATA_DIR=gs://yangyou-mnist/data
export MODEL_DIR=gs://yangyou-mnist/log
for i in 10 11 12 13 14 15
do
	for j in 0.004 0.005 0.006 0.007 0.008 0.01 0.015 0.02 0.03 0.04 0.06 0.08 0.1
	do
		gsutil -m rm -R -f $MODEL_DIR/*
		python mnist_tpu_warm.py --tpu=$TPU_NAME --use_tpu=True --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --batch_size=50000 --iterations=500 --train_steps=30 --eval_steps=10 --enable_predict=False --poly_power=1.0 --warm_up_epochs=$i --learning_rate=$j
	done
done
for i in 10 11 12 13 14 15
do
	for j in 0.004 0.005 0.006 0.007 0.008 0.01 0.015 0.02 0.03 0.04 0.06 0.08 0.1
	do
		gsutil -m rm -R -f $MODEL_DIR/*
		python mnist_tpu_warm.py --tpu=$TPU_NAME --use_tpu=True --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --batch_size=50000 --iterations=500 --train_steps=30 --eval_steps=10 --enable_predict=False --poly_power=2.0 --warm_up_epochs=$i --learning_rate=$j
	done
done
for i in 10 11 12 13 14 15
do
	for j in 0.004 0.005 0.006 0.007 0.008 0.01 0.015 0.02 0.03 0.04 0.06 0.08 0.1
	do
		gsutil -m rm -R -f $MODEL_DIR/*
		python mnist_tpu_warm.py --tpu=$TPU_NAME --use_tpu=True --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --batch_size=50000 --iterations=500 --train_steps=30 --eval_steps=10 --enable_predict=False --poly_power=0.5 --warm_up_epochs=$i --learning_rate=$j
	done
done
