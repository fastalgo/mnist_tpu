#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log/*
python mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --learning_rate=0.08
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log/*
python mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --learning_rate=0.08
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log/*
python mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --learning_rate=0.08
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log/*
python mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --learning_rate=0.08
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log/*
python mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --learning_rate=0.09
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log/*
python mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --learning_rate=0.09
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log/*
python mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --learning_rate=0.09
gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log/*
python mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000 --eval_steps=10 --enable_predict=False --learning_rate=0.09
