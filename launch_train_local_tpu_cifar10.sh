#!/bin/bash

pipenv run python main_tpu.py \
	--data-source cifar10 \
	--img-size 32 \
	--img-ch 3 \
	--num-labels 10 \
	--steps-per-loop 500 \
	--train-examples 50000 \
	--eval-examples 10000 \
	--batch-size 32 \
	--ch 16 \
	--layers 3 \
	--epoch 10000 \
	--tag cifar10 \
	--tag run-$RANDOM \
	--disable-inception-score \
	$@