#!/bin/bash

pipenv run python main_tpu.py \
	--use-tpu \
	--model-dir gs://octavian-training2/gan/mnist/model \
	--result-dir ./results \
	--data-source mnist \
	--img-size 28 \
	--img-ch 1 \
	--num-labels 10 \
	--steps-per-loop 500 \
	--train-steps 1875 \
	--eval-steps 40 \
	--batch-size 32 \
	--ch 16 \
	--layers 3 \
	--epoch 10000 \
	--tag mnist \
	--tag run-$RANDOM \
	--disable-inception-score \
	$@