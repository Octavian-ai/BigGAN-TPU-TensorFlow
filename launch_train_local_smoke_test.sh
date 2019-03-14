#!/bin/bash

pipenv run python main_tpu.py \
	--steps-per-loop 1 \
	--train-steps 1 \
	--eval-steps 1 \
	--batch-size 32 \
	--ch 8 \
	--epoch 3 \
	--tag smoketest \
	--tag run-$RANDOM \
	$@