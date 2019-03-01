
import unittest
import numpy as np
import tensorflow as tf


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
        batch_size = 16
        img_size = 128
        channels = 3

        images = np.random.random([batch_size, img_size, img_size, channels]) * 255.0
        score = calculate_inception_score(images) 

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)



   

if __name__ == "__main__":
	'''Self-testing baby'''
	unittest.main()
