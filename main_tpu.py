
from main_loop import run_main_loop

import tensorflow as tf

from BigGAN import BigGAN

import argparse
import subprocess
import os.path
import math

import logging
logger = logging.getLogger(__name__)

from utils import *
from args  import *


def main():
	args = parse_args()
	setup_logging(args)

	gan = BigGAN(args)

	if args.use_tpu:
		cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			tpu=args.tpu_name,
			zone=args.tpu_zone)
		master = cluster_resolver.get_master()
	else:
		master = ''

	tpu_run_config = tf.contrib.tpu.RunConfig(
		master=master,
		evaluation_master=master,
		model_dir=model_dir(args),
		session_config=tf.ConfigProto(
			allow_soft_placement=True, 
			log_device_placement=False),
		tpu_config=tf.contrib.tpu.TPUConfig(args.steps_per_loop,
											args.num_shards),
	)

	estimator = tf.contrib.tpu.TPUEstimator(
		model_fn=lambda features, labels, mode, params: gan.tpu_model_fn(features, labels, mode, params),
		config = tpu_run_config,
		use_tpu=args.use_tpu,
		train_batch_size=args._batch_size,
		eval_batch_size=args._batch_size,
		predict_batch_size=args._batch_size,
		params=vars(args),
	)

	run_main_loop(args, estimator)


if __name__ == '__main__':
	main()

