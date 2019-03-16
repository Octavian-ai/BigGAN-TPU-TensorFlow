#!/bin/bash

nohup pipenv run python main_tpu.py \
	--use-tpu \
	--tpu-name $TPU_NAME \
	--model-dir gs://octavian-training2/gan/cifar10/model \
	--data-dir gs://octavian-training2/gan/cifar10/data \
	--result-dir ./results \
	--data-source cifar10 \
	--img-size 32 \
	--img-ch 3 \
	--num-labels 10 \
	--steps-per-loop 500 \
	--train-examples 50000 \
	--eval-examples 10000 \
	--layers 3 \
	--epoch 200 \
	--tag cifar10 \
	--tag run-$RANDOM \
	--disable-inception-score \
	$@ &