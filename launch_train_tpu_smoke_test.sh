#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu \
	--train-input-path gs://octavian-static/download/pgan/atk-vclose-128.tfrecords \
	--model-dir gs://octavian-training2/pgan/model \
	--result-dir ./results \
	--batch-size 32  \
	--verbosity INFO \
	--steps-per-loop 200 \
	--train-steps 200 \
	--eval-steps 10 \
	--epochs 5 \
	--tag run-$RANDOM \
	$@ &
	