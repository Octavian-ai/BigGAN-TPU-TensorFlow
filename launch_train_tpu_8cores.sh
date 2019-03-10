#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu \
	--train-input-path gs://octavian-static/download/imagenet/tfr-full/train* \
	--eval-input-path gs://octavian-static/download/imagenet/tfr-full/validation* \
	--model-dir gs://octavian-training2/gan/imagenet/model \
	--result-dir ./results \
	--batch-size 128  \
	--verbosity INFO \
	--steps-per-loop 5000 \
	--train-steps 9375 \
	--eval-steps 391 \
	--epochs 1000000 \
	--tag run-$RANDOM \
	$@ &
	