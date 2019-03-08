
import tensorflow as tf
import numpy as np

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

def generic_input_fn(params, path, repeat=False):

	matching_files = tf.gfile.Glob(path)

	dataset = tf.data.TFRecordDataset(matching_files)
	dataset = dataset.map(lambda record: parse_tfrecord(params, record))
	dataset = dataset.shuffle(params['shuffle_buffer'])

	if repeat:
		dataset = dataset.repeat()

	dataset = dataset.batch(params['batch_size'], drop_remainder=True)

	return dataset

def train_input_fn(params):
	return generic_input_fn(params, params['train_input_path'], repeat=True)

def eval_input_fn(params):
	return generic_input_fn(params, params['eval_input_path'], repeat=True)

def predict_input_fn(params):
	count = max(params['sample_num'], params['batch_size'], params['inception_score_num'])

	data = np.zeros([count], dtype=np.float32)
	dataset = tf.data.Dataset.from_tensor_slices(data)
	dataset = dataset.batch(params['batch_size'], drop_remainder=True)
	return dataset


def parse_tfrecord(params, record):
	if params["tfr_format"] == 'progan':
		return parse_tfrecord_progan(params, record)
	elif params["tfr_format"] == 'inception':
		return parse_tfrecord_inception(params, True, record)
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

	empty_label = tf.constant([params['batch_size'], 1], dtype=tf.int64)

	return img, empty_label


def parse_tfrecord_inception(params, is_training, record):
	'''
	Parse the records saved using the tensorflow official inception data build

	https://github.com/tensorflow/models

	'''

	image_buffer, label, bbox, label_text = parse_example_proto(record)

	image = image_preprocessing(image_buffer, bbox, is_training, params['img_size'])
	# [batch, height, width, channels] range(-1.0,1.0)

	return image, label


