
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

import math
import os.path
import glob

from image_processing import parse_example_proto, image_preprocessing

"""

This model expects training input in the following format:

batches of images
float32 tensor of shape [batch, height, width, channels]
each value ranging between -1.0 and 1.0

The width and height of the image should match the architecture in use
e.g. 128x128, 256x256, 512x512, all with three channels

"""


# --------------------------------------------------------------------------
# Input functions for each training mode
# --------------------------------------------------------------------------

def train_input_fn(params):
	return factory_input_fn(params, is_training=True)

def eval_input_fn(params):
	return factory_input_fn(params, is_training=False)

def predict_input_fn(params):
	count = params['num_labels'] * params['num_labels']

	if params['use_inception_score']:
		count = max(count, params['inception_score_sample_size'])

	# Since we drop_remainder, we need to round up to nearest batch size
	count = math.ceil(count / params['batch_size']) * params['batch_size']

	# How many times to repeat the eye
	num_tiles = math.ceil(count / params['num_labels'])
	
	# Which labels to generate
	label_data = np.eye(params['num_labels'], dtype=np.float32)
	
	dataset = tf.data.Dataset.from_tensor_slices(label_data)
	dataset = dataset.repeat(num_tiles)
	dataset = dataset.take(count)
	dataset = dataset.batch(params['batch_size'], drop_remainder=True)
	return dataset



# --------------------------------------------------------------------------
# Train/eval data sources
# --------------------------------------------------------------------------

def tfr_input_fn(params, is_training):

	path = params['train_input_path'] if is_training else params['eval_input_path']

	matching_files = tf.gfile.Glob(path)

	dataset = tf.data.TFRecordDataset(matching_files)

	if params['take_examples'] is not None:
		dataset = dataset.take(params['take_examples'])

	dataset = dataset.map(lambda record: parse_tfrecord(params, record))
	dataset = dataset.shuffle(params['batch_size']*20)
	dataset = dataset.repeat()

	dataset = dataset.batch(params['batch_size'], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
	return dataset

def tfds_input_fn(params, dataset, is_training=True):

	dataset = tfds.load(
		name=dataset, 
		split=tfds.Split.TRAIN if is_training else tfds.Split.TEST,
		data_dir=params['data_dir'])

	if params['take_examples'] is not None:
		dataset = dataset.take(params['take_examples'])

	dataset = dataset.shuffle(params['batch_size']*20)
	dataset = dataset.repeat()
	dataset = dataset.batch(params['batch_size'], drop_remainder=True)
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

	def map_fn(features):
		image = tf.cast(features["image"], tf.float32) / 127.5 - 1

		if "label" in features:
			label = tf.one_hot(features["label"], params['num_labels'], dtype=tf.float32)
		else:
			label = tf.zeros([params['batch_size'], 1])

		return image, label

	dataset = dataset.map(map_fn)
	return dataset

def factory_input_fn(params, is_training):
	if params['data_source'] == 'tfr':
		return tfr_input_fn(params, is_training)
	elif params['data_source'] in ['mnist', 'cifar10', 'cifar100', 'lsun']:
		return tfds_input_fn(params, params['data_source'], is_training)




# --------------------------------------------------------------------------
# Parse tf records
# --------------------------------------------------------------------------

def parse_tfrecord(params, record):
	if params["tfr_format"] == 'progan':
		return parse_tfrecord_progan(params, record)
	elif params["tfr_format"] == 'inception':
		return parse_tfrecord_inception(
			params, record, 
			width=params['img_size'],
			height=params['img_size'],
			is_training=False, use_summary=params['use_summary'])
	else:
		raise NotImplementedError("Unrecognised --tfr_format")

def parse_tfrecord_progan(params, record):
	'''
	Parse the records saved using the NVIDIA ProGAN dataset_tool.py

	Data is stored as CHW uint8 with values ranging 0-255
	Size is stored beside image byte strings
	Data is stored in files with suffix -rN.tfrecords

	N = 0 is the largest size, 128x128 in my personal image build

	'''

	features = tf.parse_single_example(record, features={
		'shape': tf.FixedLenFeature([3], tf.int64),
		'data': tf.FixedLenFeature([], tf.string)})
	
	data = tf.decode_raw(features['data'], tf.uint8)

	# img = tf.reshape(data, features['shape']) # The way from ProGAN
	img = tf.reshape(data, [params['img_ch'], params['img_size'], params['img_size']])

	img = tf.transpose(img, [1,2,0]) # CHW => HWC
	img = tf.cast(img, tf.float32) / 127.5 - 1

	empty_label = tf.constant([params['batch_size'], params['num_labels']], dtype=img.dtype)

	return img, empty_label


def parse_tfrecord_inception(params, record, width, height, is_training=True, use_summary=False):
	'''
	Parse the records saved using the tensorflow official inception data build

	https://github.com/tensorflow/models

	'''

	image_buffer, label, bbox, label_text = parse_example_proto(record)

	image = image_preprocessing(image_buffer, bbox, is_training, width, height, use_summary=use_summary)
	# [batch, height, width, channels] range(-1.0,1.0)

	label_one_hot = tf.one_hot(tf.squeeze(label, axis=-1), params['num_labels'], dtype=image.dtype)

	return image, label_one_hot


