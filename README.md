# BigGAN Tensorflow TPU

Simple Tensorflow TPU implementation of ["Large Scale GAN Training for High Fidelity Natural Image Synthesis" (BigGAN)](https://arxiv.org/abs/1809.11096)

I (David Mack) have been modifying this network to allow for configuration of its self-attention, to facilitate experiments into the effectiveness of different self-attention architectures.

![main](./assets/main.png)

## Implementation notes/issues

- This is a half-finished TPU conversion of [Junho Kim's](https://github.com/taki0112/BigGAN-Tensorflow) implementation. Only 128x128 supported
- **The paper** used `orthogonal initialization`, but `I used random normal initialization.` The reason is, when using the orthogonal initialization, it did not train properly.
- I have applied a hierarchical latent space, but **not** a class embeddedding.

## Usage

### Building the data

For ImageNet, use [TensorFlow's build scripts](https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started) to create TFRecord files of your chosen image size (e.g. 128x128). `--tfr-format inception`

You can also use the data build script from [NVidia's Progressive Growing of GANs](https://github.com/tkarras/progressive_growing_of_gans). `--tfr-format progan`

### Training

You can train on a Google TPU by setting the name of your TPU as an env var and running one of the training scripts. For example,

* `TPU_NAME=node-1 pipenv run ./launch_train_tpu_b128.sh`

You need to have your training data stored on a Google cloud bucket.


## Architecture
<img src = './assets/architecture.png' width = '600px'> 

### 128x128
<img src = './assets/128.png' width = '600px'> 

### 256x256
<img src = './assets/256.png' width = '600px'> 

### 512x512
<img src = './assets/512.png' width = '600px'> 

## Contributing

You're very welcome to! Submit a PR or [contact the author(s)](https://octavian.ai)

## Authors
Junho Kim, David Mack