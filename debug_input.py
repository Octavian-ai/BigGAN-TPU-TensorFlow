
"""

This utility will run the input pipeline for the sake
of generating image summaries, for debug purposes.

(TPUEstimator does not produce summaries)


"""

import tensorflow as tf
import time

from args import *
from input import *

def input_fn(params):
	matching_files = tf.gfile.Glob(params['train_input_path'])
	dataset = tf.data.TFRecordDataset(matching_files)
	dataset = dataset.shuffle(100)
	return dataset


def model_fn(features, labels, mode, params):
	
	# Done here to get the summaries in the model_fn execution
	images, labels = parse_tfrecord_inception(params, features, is_training=False, use_summary=True)

	tf.summary.image("final_image", tf.expand_dims(images, axis=0))

	# Does nothing useful, just to run tensors through the graph
	loss = tf.reduce_mean(tf.layers.dense(images, 1))
	train_op = tf.train.AdamOptimizer().minimize(loss, tf.train.get_global_step())	

	return tf.estimator.EstimatorSpec(
		loss=loss, 
		mode=mode, 
		train_op=train_op)


def test_dataset():
	args = parse_args()
	setup_logging(args)

	params = vars(args)
	params["verbosity"] = "INFO"

	estimator = tf.estimator.Estimator(
		model_fn=model_fn, 
		params=params,
		model_dir="./model/test_dataset/"+str(time.time()))

	estimator.train(input_fn=input_fn, steps=100)
	


if __name__ == "__main__":
	test_dataset()
