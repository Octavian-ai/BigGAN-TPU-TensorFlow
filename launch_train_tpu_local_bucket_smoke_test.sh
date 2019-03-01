#!/bin/bash

pipenv run python main_tpu.py \
	--steps-per-loop 10 \
	--train-steps 10 \
	--eval-steps 3 \
	--batch-size 32 \
	--ch 8 \
	--epoch 10 \
	--model-dir gs://octavian-test/pgan/model \
	--result-dir gs://octavian-test/pgan/results \
	$@