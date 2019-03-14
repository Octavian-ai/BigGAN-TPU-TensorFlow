
import unittest
import numpy as np
import tensorflow as tf
import imageio


def _dummy_image_batch_generator(batch_size=16, img_size=128, channels=3, batches=2):
	def gen():
		for i in range(batches):
			yield np.random.random([batch_size, img_size, img_size, channels]) * 255.0

	return gen

def prefetch_inception_model():
	images = _dummy_image_batch_generator(batch_size=1)
	calculate_inception_score(images)


def calculate_inception_score(image_generator, batched=True, batch_size=None, image_size=None, channels=3):
	'''
		images:  Must be [batch, height, width, channels]. Values as float32 in range [0, 255]

		returns scores as shape [batch]
	'''

	d_generator = [image_size, image_size, channels]

	if batched:
		d_generator = [batch_size] + d_generator

	with tf.Graph().as_default():
		with tf.Session() as sess:	

			dataset = tf.data.Dataset.from_generator(image_generator, tf.float32, tf.TensorShape(d_generator))
			if not batched:
				dataset = dataset.batch(32, drop_remainder=True)

			iterator = dataset.make_one_shot_iterator()
			v_image_batch = iterator.get_next()

			if channels == 1 and v_image_batch.shape[-1] == 1:
				v_image_batch = tf.tile(v_image_batch, [1,1,1,3])

			v_images = tf.contrib.gan.eval.preprocess_image(v_image_batch)
			v_image_logits = tf.contrib.gan.eval.run_inception(v_images)

			all_image_logits = []

			while True:
				try:
					image_logits = sess.run(v_image_logits)
					all_image_logits.extend(image_logits)
				except tf.errors.OutOfRangeError:
					break

			all_image_logits = np.array(all_image_logits)

			v_image_logits_unbatched = tf.placeholder(tf.float32, [None, 1008], "image_logits_unbatched")
			v_score = tf.contrib.gan.eval.classifier_score_from_logits(v_image_logits_unbatched)

			score = sess.run(v_score, feed_dict={v_image_logits_unbatched: all_image_logits})

	return float(score)


class TestInceptionScore(unittest.TestCase):

	def test_basic(self):
		images = _dummy_image_batch_generator()
		score = calculate_inception_score(images) 

		self.assertIsInstance(score, float)
		self.assertGreater(score, 0.0)


	def test_debug(self):
		image = imageio.imread("./temp/dump.png")
		grid_n = 6
		img_size = image.shape[1] // grid_n
		img_ch = image.shape[-1]

		images = np.vsplit(image, grid_n)
		images = [np.hsplit(i, grid_n) for i in images]
		images = np.reshape(np.array(images), [grid_n*grid_n, img_size, img_size, img_ch])

		with tf.Graph().as_default():
			with tf.Session() as sess:
				v_images_placeholder = tf.placeholder(dtype=tf.float32)
				v_images = tf.contrib.gan.eval.preprocess_image(v_images_placeholder)
				v_logits = tf.contrib.gan.eval.run_inception(v_images)
				v_score = tf.contrib.gan.eval.classifier_score_from_logits(v_logits)
				score, logits = sess.run([v_score, v_logits], feed_dict={v_images_placeholder:images})


		imageio.imwrite("./temp/inception_logits.png", logits)





   

if __name__ == "__main__":
	'''Self-testing baby'''
	unittest.main()
