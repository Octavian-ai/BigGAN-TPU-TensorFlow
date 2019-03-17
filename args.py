

from comet_ml import Experiment

import tensorflow as tf

import argparse
import subprocess
import os.path
import os

import logging
import coloredlogs
logger = logging.getLogger(__name__)

from utils import *


comet_ml_api_key 	= os.environ.get("COMET_ML_API_KEY", 	None)
comet_ml_workspace 	= os.environ.get("COMET_ML_WORKSPACE", 	None)
comet_ml_project 	= os.environ.get("COMET_ML_PROJECT", 	"BigGAN")


"""parsing and configuration"""
def parse_args():
	desc = "Tensorflow implementation of BigGAN"
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--tag'              , action="append" , default=[])
	parser.add_argument('--phase'            , type=str        , default='train'                                           , help='train or test ?')
	
	parser.add_argument('--data-source'      , type=str        , default="tfr", choices=['tfr', 'mnist', 'cifar10', 'cifar100', 'lsun'], help="Where to get data from - tfrecords or mnist (internally downloaded)")
	parser.add_argument('--train-input-path' , type=str        , default='./datasets/imagenet/train*')
	parser.add_argument('--eval-input-path'  , type=str        , default='./datasets/imagenet/validate*')
	parser.add_argument('--tfr-format'       , type=str        , default='inception', choices=['inception', 'progan'], help="What format the tf records use. ProGAN is from Nvidia's github, inception is from the tensorflow/models/research github")
	parser.add_argument('--num-labels'       , type=int        , default=1000, help="Number of different possible labels")

	parser.add_argument('--model-dir'        , type=str        , default='model')
	parser.add_argument('--result-dir'       , type=str        , default='results')
	parser.add_argument('--data-dir'         , type=str        , default=None)


	# SAGAN
	# batch_size = 256
	# base channel = 64
	# epoch =  (1M iterations)
	# self-attn-res = [64]

	parser.add_argument('--img-size'        , type=int             , default=128                               , help='The width and height of the input/output image')
	parser.add_argument('--img-ch'          , type=int             , default=3                                 , help='The number of channels in the input/output image')

	parser.add_argument('--epochs'          , type=int             , default=1000000                           , help='The number of training iterations')
	parser.add_argument('--predict-every'   , type=int             , default=1                                 , help='How many training epochs to do before predicting')
	parser.add_argument('--train-examples'  , type=int             , default=1281167                           , help='The number of training examples in the dataset (used to calculate steps per epoch). Default to ImageNet values')
	parser.add_argument('--eval-examples'   , type=int             , default=50000                             , help='The number of eval examples in the dataset (used to calculate steps per epoch). Default to ImageNet values')
	parser.add_argument('--shuffle-buffer'  , type=int             , default=4000 )
	

	parser.add_argument('--batch-size'      , type=int             , default=2048  , dest="_batch_size"        , help='The size of batch across all GPUs')
	parser.add_argument('--ch'              , type=int             , default=96                                , help='base channel number per layer')
	parser.add_argument('--layers'          , type=int             , default=5 )

	parser.add_argument('--use-tpu'         , action='store_true')
	parser.add_argument('--tpu-name'        , type=str             , default=None )
	parser.add_argument('--tpu-zone'		, type=str, default='us-central1-f')
	parser.add_argument('--steps-per-loop'  , type=int             , default=100)

	parser.add_argument('--disable-comet'   , action='store_false', dest='use_comet')
	parser.add_argument('--disable-inception-score'   , action='store_false', dest='use_inception_score')
	parser.add_argument('--disable-label-cond'   , action='store_false', dest='use_label_cond')

	parser.add_argument('--enable-summary', action='store_true', dest='use_summary')

	parser.add_argument('--self-attn-res'   , action='append', default=[] )

	parser.add_argument('--g-lr'            , type=float           , default=0.00005                           , help='learning rate for generator')
	parser.add_argument('--d-lr'            , type=float           , default=0.0002                            , help='learning rate for discriminator')

	# if lower batch size
	# g_lr = 0.0001
	# d_lr = 0.0004

	# if larger batch size
	# g_lr = 0.00005
	# d_lr = 0.0002

	parser.add_argument('--beta1'          , type=float    , default=0.0           , help='beta1 for Adam optimizer')
	parser.add_argument('--beta2'          , type=float    , default=0.9           , help='beta2 for Adam optimizer')
	parser.add_argument('--moving-decay'   , type=float    , default=0.9999        , help='moving average decay for generator')

	parser.add_argument('--z-dim'          , type=int      , default=128           , help='Dimension of noise vector')
	parser.add_argument('--sn'             , type=str2bool , default=True          , help='using spectral norm')

	parser.add_argument('--gan-type'       , type=str      , default='hinge'       , choices=['gan', 'lsgan', 'wgan-gp', 'wgan-lp', 'dragan', 'hinge'])
	parser.add_argument('--ld'             , type=float    , default=10.0          , help='The gradient penalty lambda')
	parser.add_argument('--n-critic'       , type=int      , default=2             , help='The number of critic')

	# IGoodfellow says sould be 50k
	parser.add_argument('--inception-score-sample-size'     , type=int      , default=50000            , help='The number of sample images to use in inception score')
	# parser.add_argument('--num-samples'     , type=int      , default=36            , help='The number of sample images to save')
	
	parser.add_argument('--verbosity', type=str, default='INFO')

	args = parser.parse_args()
	return check_args(args)



def check_args(args):

	assert args.epochs >= 1, "number of epochs must be larger than or equal to one"
	assert args._batch_size >= 1, "batch size must be larger than or equal to one"
	assert args.ch >= 8, "--ch cannot be less than 8 otherwise some dimensions of the network will be size 0"

	if args.data_source == "mnist":
		assert args.img_ch == 1
		assert args.img_size == 28
		assert args.num_labels == 10

	if args.use_tpu:
		assert args.tpu_name is not None, "Please provide a --tpu-name"

	if args.use_comet:
		assert comet_ml_api_key is not None,   "Please provide your comet API key as $COMET_ML_API_KEY or specify --disable-comet. Comet is a cloud ML experiment visualisation platform."
		assert comet_ml_workspace is not None, "Please provide your comet API key as $COMET_ML_WORKSPACE or specify --disable-comet. Comet is a cloud ML experiment visualisation platform."

	tf.gfile.MakeDirs(suffixed_folder(args, args.result_dir))
	tf.gfile.MakeDirs("./temp/")

	return args



def model_dir(args):
	return os.path.join(args.model_dir, *args.tag, model_name(args))



def setup_logging(args):

	# Remove existing handlers at the root
	logging.getLogger().handlers = []

	coloredlogs.install(level=args.verbosity, logger=logger)

	for i in ['main_tpu', 'main_gpu', 'main_loop', 'utils', 'input', 'tensorflow', 'ops', 'BigGAN']:
		coloredlogs.install(level=args.verbosity, logger=logging.getLogger(i))

	logger.info(f"cmd args: {vars(args)}")

