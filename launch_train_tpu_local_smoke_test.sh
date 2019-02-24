#!/bin/bash

pipenv run python main_tpu.py --steps-per-loop 100 --train-steps 100 --batch-size 4 --ch 8 --epoch 3