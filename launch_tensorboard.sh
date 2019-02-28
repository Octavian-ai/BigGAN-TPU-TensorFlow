#!/bin/bash

nohup tensorboard --logdir gs://octavian-training2/pgan/model/ > nohup-tensorboard.out &
