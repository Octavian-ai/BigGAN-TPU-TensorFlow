#!/bin/bash

pipenv run python main_tpu.py --steps-per-loop 10 --train-steps 10 --batch-size 4 --ch 8 --epoch 10 --sample-num 32