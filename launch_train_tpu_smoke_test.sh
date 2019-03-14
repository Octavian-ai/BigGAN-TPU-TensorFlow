#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu \
	--train-input-path gs://octavian-static/download/imagenet/tfr-full/train* \
	--eval-input-path gs://octavian-static/download/imagenet/tfr-full/validation* \
	--model-dir gs://octavian-training2/gan/imagenet/model \
	--result-dir ./results \
	--batch-size 32  \
	--verbosity INFO \
	--steps-per-loop 200 \
	--train-examples 1281 \
	--eval-examples 50 \
	--epochs 5 \
	--tag run-$RANDOM \
	--disable-comet \
	$@ &
	