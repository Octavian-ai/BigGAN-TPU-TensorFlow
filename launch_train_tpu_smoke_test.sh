#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu \
	--train-input-path gs://octavian-static/download/imagenet/tfr-full/train* \
	--eval-input-path gs://octavian-static/download/imagenet/tfr-full/validate* \
	--model-dir gs://octavian-training2/gan/imagenet/model \
	--result-dir ./results \
	--batch-size 32  \
	--verbosity INFO \
	--steps-per-loop 200 \
	--train-steps 200 \
	--eval-steps 10 \
	--epochs 5 \
	--tag run-$RANDOM \
	$@ &
	