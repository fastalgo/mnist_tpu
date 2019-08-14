export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
export TPU_NAME=v3-8-tpu
export DATA_DIR=gs://yangyou-mnist/data
export MODEL_DIR=gs://yangyou-mnist/log
for i in 1 2 4 6 8 10
do
	for j in 0.001 0.002 0.004 0.006 0.008 0.01 0.02 0.04 0.06 0.08 0.1 0.2 0.4 0.6 0.8 1.0
	do
		gsutil -m rm -R -f $MODEL_DIR/*
		python mnist_tpu.py --tpu=$TPU_NAME --use_tpu=True --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --batch_size=50000 --iterations=500 --train_steps=30 --eval_steps=10 --enable_predict=False --warm_up_epochs=$i --learning_rate=$j
	done
done
