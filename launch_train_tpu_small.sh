#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu --tpu-name "${TPU_NAME:-node-1}" \
	--train-input-path gs://octavian-static/download/pgan/atk-vclose-128.tfrecords \
	--model-dir gs://octavian-training2/pgan/model \
	--batch-size 128  \
	--verbosity INFO \
	--steps-per-loop 300 \
	--train-steps 9000 \
	--eval-steps 10 \
	$@ &
	