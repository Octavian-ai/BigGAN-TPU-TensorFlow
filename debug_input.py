
"""

This utility will run the input pipeline for the sake
of generating image summaries, for debug purposes.

(TPUEstimator does not produce summaries)


"""

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import imageio

import time

from args import *
from input import *
from inception_score import calculate_inception_score

def input_fn(params):
	matching_files = tf.gfile.Glob(params['train_input_path'])
	dataset = tf.data.TFRecordDataset(matching_files)
	dataset = dataset.shuffle(params['batch_size'] * 10)
	dataset = dataset.take(params["inception_score_sample_size"])
	dataset = dataset.batch(params["batch_size"], drop_remainder=True)
	return dataset


def model_fn(features, labels, mode, params):

	module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/classification/1")
	height, width = hub.get_expected_image_size(module)
	
	# Done here to get the summaries in the model_fn execution
	images = tf.map_fn(
		lambda i: parse_tfrecord_inception(params, i, width, height, is_training=False, use_summary=True)[0],
		features,
		dtype=tf.float32
		)

	tf.summary.image("final_image", images)
	
	logits = module(images) # [batch_size, height, width, 3] => [batch_size, num_classes]

	# Does nothing useful, just to run tensors through the graph
	loss = tf.reduce_mean(tf.layers.dense(images, 1))
	train_op = tf.train.AdamOptimizer().minimize(loss, tf.train.get_global_step())	

	predictions =  logits

	return tf.estimator.EstimatorSpec(
		loss=loss, 
		mode=mode, 
		train_op=train_op,
		predictions=predictions,
		)


def test_dataset():
	args = parse_args()
	setup_logging(args)

	params = vars(args)
	params["verbosity"] = "INFO"
	params['inception_score_sample_size'] = 50000
	params["batch_size"] = 128

	try:
		raise Exception("Force build")
		logits = np.loadtxt("./temp/inception_logits.txt.gz")
	except:
		estimator = tf.estimator.Estimator(
			model_fn=model_fn, 
			params=params,
			model_dir="./model/test_dataset/"+str(time.time()))

		estimator.train(input_fn=input_fn, steps=3)

		predictions = estimator.predict(input_fn=input_fn)

		logits = []
		for i in predictions:
			logits.append(i)

		logits = np.array(logits)
		np.savetxt("./temp/inception_logits.txt.gz", logits)


	print(f"Logits shape {logits.shape}")
	imageio.imwrite("./temp/inception_logits.png", logits)
	marginal_logits = np.sum(logits, axis=0, keepdims=True)
	marginal_logits = np.tile(marginal_logits, [500, 1])
	imageio.imwrite("./temp/inception_marginal_logits.png", marginal_logits)



	with tf.Graph().as_default():
		with tf.Session() as sess:
			v_logits = tf.placeholder(tf.float32, logits.shape)	
			v_score = tf.contrib.gan.eval.classifier_score_from_logits(v_logits)

			score = sess.run(v_score, feed_dict={v_logits:logits})
			score = float(score)

	# score = calculate_inception_score(lambda: (i for i in sample_images), batched=False)
	print(f"Inception score {score} using {len(logits)} samples")



if __name__ == "__main__":
	test_dataset()
