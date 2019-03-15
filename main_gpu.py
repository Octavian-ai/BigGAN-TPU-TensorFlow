

import tensorflow as tf

from BigGAN import BigGAN
from inception_score import prefetch_inception_model

import argparse
import subprocess
import os.path
import math

import logging
logger = logging.getLogger(__name__)

from utils import *
from args  import *
from main_loop import run_main_loop


def main():
	args = parse_args()
	setup_logging(args)

	gan = BigGAN(args)

	params = vars(args)
	params['batch_size'] = params['_batch_size']

	estimator = tf.estimator.Estimator(
		model_fn=lambda features, labels, mode, params: gan.gpu_model_fn(features, labels, mode, params),
		params=vars(args),
		model_dir=model_dir(args)
	)

	run_main_loop(args, estimator)


if __name__ == '__main__':
	main()

