import scipy.misc
import numpy as np
import os
import itertools
from glob import glob
import imageio
import math
import os.path
import tensorflow as tf
import tensorflow.contrib.slim as slim

from inception_score import calculate_inception_score

import logging
logger = logging.getLogger(__name__)


class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


# --------------------------------------------------------------------------
# Main loop helpers
# --------------------------------------------------------------------------

def str2bool(x):
    return x.lower() in ('true')

def model_name(args):
    mn = ""

    for i in args.self_attn_res:
        mn += f"sa{i}_"

    if args.sn :
        sn = '_sn'
    else :
        sn = ''

    mn += "{}_w{}_bs{}_ch{}_z{}{}".format(
         args.gan_type, args.img_size, args._batch_size, args.ch, args.z_dim, sn)

    return mn



def suffixed_folder(args, dir):
    return os.path.join(dir, *args.tag, model_name(args))

def imwrite(file, data):
    d = data * 255
    d = d.astype(np.uint8)
    imageio.imwrite(file, d, format="png")


def save_predictions(args, result_dir, eval_file, predictions, epoch, total_steps, experiment):

    image_frame_dim = int(np.floor(np.sqrt(args.num_samples)))
    samples = []
    labels = []

    try:
        for ct, i in enumerate(predictions):
            if ct >= args.num_samples:
                break
            samples.append(i['fake_image'])
            labels.append(i['labels'])
    
    except tf.errors.OutOfRangeError:
        pass


    if len(samples) == 0:
        logger.warning(f"No predictions returned from TensorFlow in epoch {epoch}")
        return

    else:
        logger.info(f"Generated {len(samples)} samples")

    samples = np.array(samples)
    grid_samples = samples[:image_frame_dim * image_frame_dim, :, :, :]
    grid_image = merge(inverse_transform(grid_samples), [image_frame_dim, image_frame_dim])
  
    for filename in ['epoch%02d' % epoch + '_sample.png', 'latest_sample.png']:
        file_path = os.path.join(result_dir, filename)
        with tf.gfile.Open(file_path, 'wb') as file:
            imwrite(file, grid_image)

    labelled_samples = zip(samples,labels)

    for ct, (sample, label) in  enumerate(itertools.islice(labelled_samples, args.num_labels)):
        filename = 'epoch%02d' % epoch +  f"_sample_{ct}_label_{label}.png"
        file_path = os.path.join(result_dir, filename)
        with tf.gfile.Open(file_path, 'wb') as file:
            imwrite(file, sample)

        if args.use_comet:
            tmp_file_path = f"./temp/latest_label_{label}.png"
            imwrite(tmp_file_path, sample)
            experiment.log_image(tmp_file_path)


    if args.use_comet:
        tmp_file_path = "./temp/latest_sample.png"
        imwrite(tmp_file_path, grid_image)
        experiment.log_image(tmp_file_path)

    

    if args.use_inception_score:

        def sample_gen():
            for i in samples:
                yield i

        inception_score = calculate_inception_score(sample_gen, batched=False, channels=args.img_ch)
        inception_score_dict = {'inception_score': inception_score}

        logger.info(f"step {total_steps}\t{inception_score_dict}")

        if args.use_comet:
            experiment.log_metric('inception_score', inception_score)
        
        eval_file.write(f"Step {total_steps}\t inception_score={inception_score} inception_score_sample_size={len(samples)}\n")





def save_evaluation(args, eval_file, evaluation, epoch, total_steps):
    eval_file.write(f"Step {total_steps}\t{evaluation}\n")


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def inverse_transform(images):
    return (images+1.)/2.


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)



##################################################################################
# Regularization
##################################################################################

def orthogonal_regularizer(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

def orthogonal_regularizer_fully(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully