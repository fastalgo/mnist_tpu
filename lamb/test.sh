export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log3/*
#python mnist_tpu.py --tpu=infer1 --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log3/ --batch_size=1024 --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --warm_up_epochs=0 --weight_decay_input=0.01 --learning_rate=0.001
python mnist_tpu.py --tpu=infer1 --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log3/ --batch_size=1024 --iterations=500 --train_steps=1758 --eval_steps=10 --enable_predict=False --warm_up_epochs=1 --learning_rate=0.004
