# BigGAN-Tensorflow-TPU

**This is a half-finished TPU conversion of Junho Kim's implementation. Only 128x128 supported ATM**

Simple Tensorflow implementation of ["Large Scale GAN Training for High Fidelity Natural Image Synthesis" (BigGAN)](https://arxiv.org/abs/1809.11096)

![main](./assets/main.png)

## Issue
* **The paper** used `orthogonal initialization`, but `I used random normal initialization.` The reason is, when using the orthogonal initialization, it did not train properly.
* I have applied a hierarchical latent space, but **not** a class embeddedding.

## Usage

### train
* pipenv run ./launch_tpu_8.sh


## Architecture
<img src = './assets/architecture.png' width = '600px'> 

### 128x128
<img src = './assets/128.png' width = '600px'> 

### 256x256
<img src = './assets/256.png' width = '600px'> 

### 512x512
<img src = './assets/512.png' width = '600px'> 

## Author
Junho Kim
David Mack