#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
export TPU_NAME=trump
export DATA_DIR=gs://yangyou-mnist/data
export MODEL_DIR=gs://yangyou-mnist/log
gsutil -m rm -R -f $MODEL_DIR/*
python mnist_tpu.py --tpu=$TPU_NAME --use_tpu=True --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --batch_size=1024 --iterations=5000 --train_steps=2000 --eval_steps=4 --enable_predict=False --learning_rate=0.2
