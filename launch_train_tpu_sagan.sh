#!/bin/bash

nohup pipenv run python3.6 main_tpu.py --use-tpu \
	--train-input-path gs://octavian-static/download/imagenet/tfr-full/train* \
	--eval-input-path gs://octavian-static/download/imagenet/tfr-full/validation* \
	--model-dir gs://octavian-training2/gan/imagenet/model \
	--result-dir ./results \
	--batch-size 256  \
	--ch 64 \
	--self-attn-res 64 \
	--g-lr 0.0001 \
	--d-lr 0.0004 \
	--verbosity INFO \
	--train-steps 9375 \
	--eval-steps 391 \
	--tag sagan \
	--tag run-$RANDOM \
	$@ &
	