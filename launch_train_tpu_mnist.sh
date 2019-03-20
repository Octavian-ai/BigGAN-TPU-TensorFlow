#!/bin/bash

nohup pipenv run python main_tpu.py \
	--use-tpu \
	--tpu-name $TPU_NAME \
	--model-dir gs://octavian-training2/gan/mnist/model \
	--data-dir gs://octavian-training2/gan/mnist/data \
	--result-dir ./results \
	--data-source mnist \
	--img-size 28 \
	--img-ch 1 \
	--num-labels 10 \
	--steps-per-loop 500 \
	--train-examples 60000 \
	--eval-examples 10000 \
	--layers 3 \
	--epoch 20 \
	--tag mnist \
	--tag run-$RANDOM \
	$@ &