#export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"
#python /usr/share/models/official/mnist/mnist_tpu.py --tpu=infer --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log/ --iterations=500 --train_steps=2000
for i in 0
do
	for j in 0.01 0.02 0.04 0.06 0.08 0.1 0.001 0.002 0.004 0.006 0.008 0.0001 0.0002 0.0004 0.0006 0.0008 0.00001 0.00002 0.00004 0.00006 0.00008 0.1
	do
		echo "warmup $i, weight decay $j"
		gsutil -m rm -R -f gs://bert-pretrain-data/mnist_log2/*
		python mnist_tpu.py --tpu=infer2 --use_tpu=True --data_dir=gs://bert-pretrain-data/mnist/ --model_dir=gs://bert-pretrain-data/mnist_log2/ --batch_size=60000 --iterations=500 --train_steps=35 --eval_steps=10 --enable_predict=False --warm_up_epochs=$i --weight_decay_input=0.01 --learning_rate=$j
	done
done
