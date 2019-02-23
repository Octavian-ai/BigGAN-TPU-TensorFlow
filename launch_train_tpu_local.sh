#!/bin/bash

pipenv run python main_tpu.py --steps-per-loop 10 --train-max-steps 20 --batch-size 32  