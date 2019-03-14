#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu \
	--train-input-path gs://octavian-static/download/imagenet/tfr-full/train* \
	--eval-input-path gs://octavian-static/download/imagenet/tfr-full/validation* \
	--model-dir gs://octavian-training2/gan/imagenet/model \
	--result-dir ./results \
	--batch-size 256  \
	--steps-per-loop 100 \
	--ch 64 \
	--layers 3 \
	--img-size 32 \
	--self-attn-res 16 \
	--g-lr 0.0001 \
	--d-lr 0.0004 \
	--verbosity INFO \
	--train-examples 1281167 \
	--eval-examples 50000 \
	--tag sagan-sm \
	--tag run-$RANDOM \
	$@ &
	