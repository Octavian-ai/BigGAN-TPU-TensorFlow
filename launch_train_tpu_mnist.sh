#!/bin/bash

nohup pipenv run python main_tpu.py \
	--use-tpu \
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
	--batch-size 64 \
	--ch 64 \
	--layers 3 \
	--self-attn-res 16 \
	--epoch 20 \
	--tag mnist \
	--tag run-$RANDOM \
	--disable-inception-score \
	$@ &