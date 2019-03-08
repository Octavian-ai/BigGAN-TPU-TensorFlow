#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu \
	--train-input-path gs://octavian-static/download/imagenet/tfr-full/train* \
	--eval-input-path gs://octavian-static/download/imagenet/tfr-full/validation* \
	--model-dir gs://octavian-training2/gan/imagenet/model \
	--result-dir ./results \
	--batch-size 128  \
	--verbosity INFO \
	--steps-per-loop 500 \
	--train-steps 2000 \
	--eval-steps 10 \
	--epochs 20 \
	--tag run-$RANDOM \
	$@ &
	