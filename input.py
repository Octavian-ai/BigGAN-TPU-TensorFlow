
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

class InputPipeline(object):
	# Extra optimized
	# From https://cloud.google.com/tpu/docs/inception-v3-advanced

	def __init__(self, is_training, path):
		self.path = path
		self.is_training = is_training

	def __call__(self, params):
		# Storage
		dataset = tf.data.Dataset.list_files(self.path)
		if self.is_training and params['initial_shuffle_buffer_size'] > 0:
			dataset = dataset.shuffle(buffer_size=params['initial_shuffle_buffer_size'])
		if self.is_training:
			dataset = dataset.repeat()

		def prefetch_dataset(filename):
			dataset = tf.data.TFRecordDataset(filename, buffer_size=params['prefetch_dataset_buffer_size'])
			return dataset

		dataset = dataset.apply(
				tf.contrib.data.parallel_interleave(
						prefetch_dataset,
						cycle_length=params['num_files_infeed'],
						sloppy=True))
		if params['followup_shuffle_buffer_size'] > 0:
			dataset = dataset.shuffle(
				buffer_size=params['followup_shuffle_buffer_size'])

		# Preprocessing
		dataset = dataset.map(
				lambda record: parse_tfrecord(params, record),
				num_parallel_calls=params['num_parallel_calls'])

		dataset = dataset.prefetch(batch_size)
		dataset = dataset.apply(
				tf.contrib.data.batch_and_drop_remainder(batch_size))
		dataset = dataset.prefetch(2)	# Prefetch overlaps in-feed with training
		images, labels = dataset.make_one_shot_iterator().get_next()

		# Transfer
		return images, labels


def generic_input_fn(params, path, repeat=False):

	matching_files = tf.gfile.Glob(path)

	dataset = tf.data.TFRecordDataset(matching_files)
	dataset = dataset.map(lambda record: parse_tfrecord(params, record))
	dataset = dataset.shuffle(params['batch_size']*20)

	if repeat:
		dataset = dataset.repeat()

	dataset = dataset.batch(params['batch_size'], drop_remainder=True)

	return dataset

def train_input_fn(params):
	return generic_input_fn(params, params['train_input_path'], repeat=True)

def eval_input_fn(params):
	return generic_input_fn(params, params['eval_input_path'], repeat=True)

def predict_input_fn(params):
	count = max(params['num_samples'], params['batch_size'], params['inception_score_sample_size'])
	
	# Which labels to generate
	label_data = np.eye(params['num_labels'], dtype=np.float32)
	label_data = np.repeat(label_data, 1 + (count // params['num_labels']), axis=0)
	np.random.shuffle(label_data)
	label_data = label_data[:count]

	dataset = tf.data.Dataset.from_tensor_slices(label_data)
	dataset = dataset.batch(params['batch_size'], drop_remainder=True)
	return dataset


def parse_tfrecord(params, record):
	if params["tfr_format"] == 'progan':
		return parse_tfrecord_progan(params, record)
	elif params["tfr_format"] == 'inception':
		return parse_tfrecord_inception(
			params, record, 
			width=params['img_size'],
			height=params['img_size'],
			is_training=False, use_summary=False)
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

	empty_label = tf.constant([params['batch_size'], params['num_labels']], dtype=tf.int64)

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


