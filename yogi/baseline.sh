export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
export TPU_NAME=v3-8-4tb
export DATA_DIR=gs://bert-pretrain-data/mnist
export MODEL_DIR=gs://bert-pretrain-data/mnist_log
for i in 1
do
	for j in 0.001
	do
		gsutil -m rm -R -f $MODEL_DIR/*
		python mnist_tpu.py --tpu=$TPU_NAME --use_tpu=True --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --batch_size=1024 --iterations=2000 --train_steps=1600 --eval_steps=10 --enable_predict=False --warm_up_epochs=$i --learning_rate=$j
	done
done
