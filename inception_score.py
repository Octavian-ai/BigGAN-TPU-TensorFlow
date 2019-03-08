
import unittest
import numpy as np
import tensorflow as tf
import imageio


def _generate_dummy_images(batch_size = 16, img_size = 128, channels = 3):
	return np.random.random([batch_size, img_size, img_size, channels]) * 255.0

def prefetch_inception_model():
	images = _generate_dummy_images(batch_size=1)
	calculate_inception_score(images)


def calculate_inception_score(images):
	'''
		images:  Must be [batch, height, width, channels]. Values as float32 in range [0, 255]

		returns scores as shape [batch]
	'''

	with tf.Graph().as_default():
		with tf.Session() as sess:	
			v_images_placeholder = tf.placeholder(dtype=tf.float32)
			v_images = tf.contrib.gan.eval.preprocess_image(v_images_placeholder)
			v_score = tf.contrib.gan.eval.classifier_score(v_images, tf.contrib.gan.eval.run_inception)
			score = sess.run(v_score, feed_dict={v_images_placeholder:images})

	return float(score)


class TestInceptionScore(unittest.TestCase):

	def test_basic(self):
		images = _generate_dummy_images()
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
