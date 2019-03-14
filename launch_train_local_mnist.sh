#!/bin/bash

pipenv run python main_tpu.py \
	--data-source mnist \
	--img-size 28 \
	--img-ch 1 \
	--num-labels 10 \
	--steps-per-loop 1 \
	--train-steps 1 \
	--eval-steps 1 \
	--batch-size 32 \
	--ch 8 \
	--layers 3 \
	--epoch 3 \
	--tag mnist \
	--tag run-$RANDOM \
	$@