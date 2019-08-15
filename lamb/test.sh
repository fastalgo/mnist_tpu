export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
export TPU_NAME=v3-8-tpu
export DATA_DIR=gs://yangyou-mnist/data
export MODEL_DIR=gs://yangyou-mnist/log
for i in 8 9 10 11 12 13 14 15
do
	for j in 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16
	do
		gsutil -m rm -R -f $MODEL_DIR/*
		python mnist_tpu.py --tpu=$TPU_NAME --use_tpu=True --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --batch_size=50000 --iterations=500 --train_steps=30 --eval_steps=10 --enable_predict=False --warm_up_epochs=$i --learning_rate=$j
	done
done
