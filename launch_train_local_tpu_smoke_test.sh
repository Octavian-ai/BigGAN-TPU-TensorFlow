#!/bin/bash

pipenv run python main_tpu.py \
	--data-source mnist \
	--img-size 28 \
	--img-ch 1 \
	--num-labels 10 \
	--steps-per-loop 1 \
	--train-examples 32 \
	--eval-examples 32 \
	--batch-size 32 \
	--inception-score-sample-size 36 \
	--ch 8 \
	--layers 3 \
	--epoch 1 \
	--tag smoketest \
	--tag run-$RANDOM \
	--disable-comet \
	$@