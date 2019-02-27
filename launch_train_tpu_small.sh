#!/bin/bash

python3.6 main_tpu.py --use-tpu --tpu-name node-1 \
	--train-input-path gs://octavian-static/download/pgan/atk-vclose-128.tfrecords \
	--model-dir gs://octavian-training2/pgan/model \
	--batch-size 16  \
	--verbosity INFO \
	--steps-per-loop 10 \
	--train-steps 10 \
	--eval-steps 10 \

