import time
from ops import *
from utils import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
from tensorflow.contrib.opt import MovingAverageOptimizer


class BigGAN_256(object):


    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            # 7
            if self.z_dim == 128:
                split_dim = 18
                split_dim_remainder = self.z_dim - (split_dim * 6)

                z_split = tf.split(z, num_or_size_splits=[split_dim] * 6 + [split_dim_remainder], axis=-1)

            else:
                split_dim = self.z_dim // 7
                split_dim_remainder = self.z_dim - (split_dim * 7)

                if split_dim_remainder == 0 :
                    z_split = tf.split(z, num_or_size_splits=[split_dim] * 7, axis=-1)
                else :
                    z_split = tf.split(z, num_or_size_splits=[split_dim] * 6 + [split_dim_remainder], axis=-1)


            ch = 16 * self.ch
            x = fully_conneted(z_split[0], units=4 * 4 * ch, sn=self.sn, scope='dense')
            x = tf.reshape(x, shape=[-1, 4, 4, ch])

            x = resblock_up_condition(x, z_split[1], channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_up_16')
            ch = ch // 2

            x = resblock_up_condition(x, z_split[2], channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_up_8_0')
            x = resblock_up_condition(x, z_split[3], channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_up_8_1')
            ch = ch // 2

            x = resblock_up_condition(x, z_split[4], channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_up_4')
            ch = ch // 2

            x = resblock_up_condition(x, z_split[5], channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_up_2')

            # Non-Local Block
            x = self_attention_2(x, channels=ch, sn=self.sn, scope='self_attention')
            ch = ch // 2

            x = resblock_up_condition(x, z_split[6], channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_up_1')

            x = batch_norm(x, is_training)
            x = relu(x)
            x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='G_logit')

            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = self.ch

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_down_1')
            ch = ch * 2

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_down_2')

            # Non-Local Block
            x = self_attention_2(x, channels=ch, sn=self.sn, scope='self_attention')
            ch = ch * 2

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_down_4')
            ch = ch * 2

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_down_8_0')
            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_down_8_1')
            ch = ch * 2

            x = resblock_down(x, channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock_down_16')

            x = resblock(x, channels=ch, use_bias=False, is_training=is_training, sn=self.sn, scope='resblock')
            x = relu(x)

            x = global_sum_pooling(x)

            x = fully_conneted(x, units=1, sn=self.sn, scope='D_logit')

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

    def build_model(self):
        """ Graph Input """
        # images
        Image_Data_Class = ImageData(self.img_size, self.c_dim, self.custom_dataset)
        inputs = tf.data.Dataset.from_tensor_slices(self.data)

        gpu_device = '/gpu:0'
        inputs = inputs.\
            apply(shuffle_and_repeat(self.dataset_num)).\
            apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))

        inputs_iterator = inputs.make_one_shot_iterator()

        self.inputs = inputs_iterator.get_next()

        # noises
        self.z = tf.truncated_normal(shape=[self.batch_size, 1, 1, self.z_dim], name='random_z')

        """ Loss Function """
        # output of D for real images
        real_logits = self.discriminator(self.inputs)

        # output of D for fake images
        fake_images = self.generator(self.z)
        fake_logits = self.discriminator(fake_images, reuse=True)

        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
            GP = self.gradient_penalty(real=self.inputs, fake=fake_images)
        else:
            GP = 0

        # get loss for discriminator
        self.d_loss = discriminator_loss(self.gan_type, real=real_logits, fake=fake_logits) + GP

        # get loss for generator
        self.g_loss = generator_loss(self.gan_type, fake=fake_logits)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)

            self.opt = MovingAverageOptimizer(tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2), average_decay=self.moving_decay)

            self.g_optim = self.opt.minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)


