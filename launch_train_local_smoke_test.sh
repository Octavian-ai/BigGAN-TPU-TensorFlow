#!/bin/bash

pipenv run python main_tpu.py \
	--steps-per-loop 1 \
	--train-examples 32 \
	--eval-examples 32 \
	--batch-size 32 \
	--ch 8 \
	--epoch 1 \
	--tag smoketest \
	--tag run-$RANDOM \
	--disable-comet \
	$@