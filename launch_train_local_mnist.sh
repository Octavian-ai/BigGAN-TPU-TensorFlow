#!/bin/bash

pipenv run python main_tpu.py \
	--data-source mnist \
	--img-size 28 \
	--img-ch 1 \
	--num-labels 10 \
	--steps-per-loop 500 \
	--train-examples 60000 \
	--eval-examples 10000 \
	--batch-size 32 \
	--ch 16 \
	--layers 3 \
	--epoch 10000 \
	--tag mnist \
	--tag run-$RANDOM \
	--disable-inception-score \
	$@