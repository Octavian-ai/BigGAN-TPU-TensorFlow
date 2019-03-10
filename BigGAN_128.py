import time
from ops import *
from utils import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
from tensorflow.contrib.opt import MovingAverageOptimizer

import logging
logger = logging.getLogger(__name__)

class BigGAN_128(object):

	def __init__(self, args):
		pass


	##################################################################################
	# Generator
	##################################################################################

	def generator(self, params, z, labels, is_training=True, reuse=False):
		logger.debug("generator")
		with tf.variable_scope("generator", reuse=reuse):
			# 6
			if params['z_dim'] == 128:
				split_dim = 20
				split_dim_remainder = params['z_dim'] - (split_dim * 5)

				z_split = tf.split(z, num_or_size_splits=[split_dim] * 5 + [split_dim_remainder], axis=-1)

			else:
				split_dim = params['z_dim'] // 6
				split_dim_remainder = params['z_dim'] - (split_dim * 6)

				if split_dim_remainder == 0 :
					z_split = tf.split(z, num_or_size_splits=[split_dim] * 6, axis=-1)
				else :
					z_split = tf.split(z, num_or_size_splits=[split_dim] * 5 + [split_dim_remainder], axis=-1)

			ch = 16 * params['ch']
			sn = params['sn']

			x = fully_connected(z_split[0], units=4 * 4 * ch, sn=sn, scope='dense')
			x = tf.reshape(x, shape=[-1, 4, 4, ch])

			for i in range(params['layers']):
				x_size = x.shape[-2]
				cond = tf.concat([z_split[i], labels], axis=-1)
				x = resblock_up_condition(x, cond, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope=f"resblock_up_w{x_size}_ch{ch//params['ch']}")
				
				x_size = x.shape[-2]
				if x_size in params['self_attn_res']:
					x = self_attention_2(x, channels=ch, sn=sn, scope=f"self_attention_w{x_size}_ch{ch//params['ch']}")

				ch = ch // 2

			ch = ch * 2

			x = batch_norm(x, is_training)
			x = relu(x)
			x = conv(x, channels=params['img_ch'], kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='G_logit')

			x = tanh(x)

			logger.debug("--")

			return x

	##################################################################################
	# Discriminator
	##################################################################################

	def discriminator(self, params, x, is_training=True, reuse=False):
		logger.debug("discriminator")
		with tf.variable_scope("discriminator", reuse=reuse):
			ch = params['ch']
			sn = params['sn']

			for i in range(params['layers']):

				x_size = x.shape[-2]
				x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope=f"resblock_down_w{x_size}_ch{ch//params['ch']}")

				x_size = x.shape[-2]
				if x_size in params['self_attn_res']:
					x = self_attention_2(x, channels=ch, sn=sn, scope=f"self_attention_w{x_size}_ch{ch//params['ch']}")
				
				ch = ch * 2

			ch = ch // 2

			x_size = x.shape[-2]
			x = resblock(x, channels=ch, use_bias=False, is_training=is_training, sn=sn, scope=f"resblock_w{x_size}_ch{ch//params['ch']}")
			x = relu(x)

			x = global_sum_pooling(x)

			x = fully_connected(x, units=1, sn=sn, scope='D_logit')

			logger.debug("--")

			return x

	def gradient_penalty(self, real, fake):
		if self.gan_type.__contains__('dragan'):
			eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
			_, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
			x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

			fake = real + 0.5 * x_std * eps

		alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
		interpolated = real + alpha * (fake - real)

		logit = self.discriminator(interpolated, reuse=True)

		grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
		grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

		GP = 0

		# WGAN - LP
		if self.gan_type == 'wgan-lp':
			GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

		elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
			GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

		return GP

	##################################################################################
	# Model
	##################################################################################

	def base_model_fn(self, features, labels, mode, params):

		# Because we cannot pass in labels in predict mode (despite them being useful 
		# for GANs), I've passed the labels in as the (otherwise unneeded) features
		# it's a bit of a hack, sorry.
		if mode == tf.estimator.ModeKeys.PREDICT:
			labels = features
		
		# Latent input to generate images
		z = tf.truncated_normal(shape=[params.batch_size, 1, 1, params.z_dim], name='random_z')
		
		# Conditioning of the batch normalization based on image label
		labels_expanded = tf.expand_dims(labels, 1)
		labels_expanded = tf.expand_dims(labels_expanded, 1)

		# generate and critique fake images
		fake_images = self.generator(params, z, labels_expanded)
		fake_logits = self.discriminator(params, fake_images)
		g_loss = generator_loss(params.gan_type, fake=fake_logits)

		# Train the discriminator
		if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
			real_logits = self.discriminator(params, features, reuse=True)

			if params.gan_type.__contains__('wgan') or params.gan_type == 'dragan':
				GP = self.gradient_penalty(real=features, fake=fake_images)
			else:
				GP = 0

			d_loss = discriminator_loss(params.gan_type, real=real_logits, fake=fake_logits) + GP

		else:
			d_loss = 0

		# Get all the vars
		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'discriminator' in var.name]
		g_vars = [var for var in t_vars if 'generator' in var.name]

		return d_loss, d_vars, g_loss, g_vars, fake_images, fake_logits, z


	def tpu_model_fn(self, features, labels, mode, params):

		params = EasyDict(**params)

		d_loss, d_vars, g_loss, g_vars, fake_images, fake_logits, z = self.base_model_fn(features, labels, mode, params)

		# --------------------------------------------------------------------------
		# Predict
		# --------------------------------------------------------------------------

		if mode == tf.estimator.ModeKeys.PREDICT:
		    predictions = {
				"z": z,
				"fake_image": fake_images,
				"fake_logits": fake_logits,
		    }
		    return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)


		# --------------------------------------------------------------------------
		# Train or Eval
		# --------------------------------------------------------------------------
	
		loss = g_loss
		for i in range(params.n_critic):
			loss += d_loss

		
		if mode == tf.estimator.ModeKeys.EVAL:

			# Hack to allow it out of a fixed batch size TPU
			d_loss_batched = tf.tile(tf.expand_dims(d_loss, 0), [params.batch_size])
			g_loss_batched = tf.tile(tf.expand_dims(g_loss, 0), [params.batch_size])

			d_grad = tf.gradients(d_loss, d_vars)
			g_grad = tf.gradients(g_loss, g_vars)

			d_grad_joined = tf.concat([
				tf.reshape(i, [-1]) for i in d_grad
			], axis=-1)

			g_grad_joined = tf.concat([
				tf.reshape(i, [-1]) for i in g_grad
			], axis=-1)

			def tpu_metric_fn(d_loss, g_loss, fake_logits, d_grad, g_grad):
				return {
					"d_loss"      : tf.metrics.mean(d_loss),
					"g_loss"      : tf.metrics.mean(g_loss),
					"fake_logits" : tf.metrics.mean(fake_logits),
					"d_grad"      : tf.metrics.mean(d_grad),
					"g_grad"      : tf.metrics.mean(g_grad),
				}

			return tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=loss, 
				eval_metrics=(
					tpu_metric_fn, 
					[d_loss_batched, g_loss_batched, fake_logits, d_grad_joined, g_grad_joined]
				)
			)

		
		if mode == tf.estimator.ModeKeys.TRAIN:

			# Create training ops for both D and G

			d_optimizer = tf.train.AdamOptimizer(params.d_lr, beta1=params.beta1, beta2=params.beta2)
			
			if params.use_tpu:
				d_optimizer = tf.contrib.tpu.CrossShardOptimizer(d_optimizer)

			d_train_op = d_optimizer.minimize(d_loss, var_list=d_vars, global_step=tf.train.get_global_step())

			
			g_optimizer = MovingAverageOptimizer(
				tf.train.AdamOptimizer(params.g_lr, beta1=params.beta1, beta2=params.beta2), 
				average_decay=params.moving_decay)
			
			if params.use_tpu:
				g_optimizer = tf.contrib.tpu.CrossShardOptimizer(g_optimizer)

			g_train_op = g_optimizer.minimize(g_loss, var_list=g_vars, global_step=tf.train.get_global_step())


			# For each training op of G, do n_critic training ops of D
			train_ops = [g_train_op]
			for i in range(params.n_critic):
				train_ops.append(d_train_op)
			train_op = tf.group(*train_ops)

			return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)


