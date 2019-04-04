#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
for i in 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 9 8 7 6 5
do
	for j in 0.001 0.002 0.005 0.006 0.008 0.01 0.02 0.04 0.06 0.08 0.1
	do
		echo "warmup $i, LR $j"
		gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log3/*
		python mnist_tpu.py --tpu=yangyouucb --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log3/ --batch_size=60000 --iterations=500 --train_steps=35 --eval_steps=10 --enable_predict=False --warm_up_epochs=$i --learning_rate=$j
	done
done
