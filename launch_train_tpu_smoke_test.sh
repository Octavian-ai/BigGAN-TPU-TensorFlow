#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu \
	--train-input-path gs://octavian-static/download/imagenet/tfr-full/train* \
	--eval-input-path gs://octavian-static/download/imagenet/tfr-full/validation* \
	--model-dir gs://octavian-training2/gan/imagenet/model \
	--result-dir ./results \
	--batch-size 32  \
	--verbosity INFO \
	--steps-per-loop 1 \
	--train-examples 32 \
	--eval-examples 32 \
	--epochs 1 \
	--tag run-$RANDOM \
	--disable-comet \
	$@ &
	