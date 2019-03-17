#!/bin/bash

nohup pipenv run python main_tpu.py \
	--use-tpu \
	--tpu-name $TPU_NAME \
	--data-source cifar10 \
	--model-dir gs://octavian-training2/gan/cifar10/model \
	--data-dir gs://octavian-training2/gan/cifar10/data \
	--train-examples 50000 \
	--eval-examples 10000 \
	--img-size 32 \
	--img-ch 3 \
	--num-labels 10 \
	--layers 3 \
	--batch-size 512 \
	--ch 96 \
	--epoch 200000 \
	--predict-every 40 \
	--tag cifar10 \
	--tag run-$RANDOM \
	--steps-per-loop 500 \
	--disable-inception-score \
	$@ &