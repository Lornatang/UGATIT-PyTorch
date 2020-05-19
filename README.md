# UGATIT-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation
](http://arxiv.org/abs/1907.10830).

### Table of contents

1. [About U-GAT-IT](#about-u-gat-it)
2. [Model Description](#model-description)
3. [Installation](#installation)
   * [Clone and install requirements](#clone-and-install-requirements)
   * [Download pretrained weights](#download-pretrained-weights)
   * [Download dataset](#download-dataset)
4. [Test](#test-eg-selfie2anime)
5. [Train](#train)
   * [Example](#example-eg-selfie2anime)
6. [Contributing](#contributing) 
7. [Credit](#credit)

### About U-GAT-IT

If you're new to U-GAT-IT, here's an abstract straight from the paper:

We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module 
and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus 
on more important regions distinguishing between source and target domains based on the attention map obtained 
by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes 
between domains, our model can translate both images requiring holistic changes and images requiring large shape 
changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided 
model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. 
Experimental results show the superiority of the proposed method compared to the existing state-of-the-art 
models with a fixed network architecture and hyper-parameters.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. 
It receives a random noise z and generates images from this noise, which is called G(z).Discriminator 
is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, 
and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, 
and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/UGATIT-PyTorch
$ cd UGATIT-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights

```bash
$ cd weights/
$ bash download_weights.sh
```

#### Download dataset

```bash
$ cd data/
$ bash get_dataset.sh
```

### Test (e.g selfie2anime)

Using pre training model to generate pictures.

```bash
$ python3 test.py --cuda
```

### Train

```text
usage: train.py [-h] [--dataroot DATAROOT] [--dataset DATASET] [--light]
                [--epochs N] [--image-size IMAGE_SIZE]
                [--decay_epochs DECAY_EPOCHS] [-b N] [--lr LR]
                [--weight-decay WEIGHT_DECAY] [-p N] [--cuda]
                [--netG_A2B NETG_A2B] [--netG_B2A NETG_B2A] [--netD_A NETD_A]
                [--netD_B NETD_B] [--netL_A NETL_A] [--netL_B NETL_B]
                [--outf OUTF] [--manualSeed MANUALSEED]

```

#### Example (e.g selfie2anime)

```bash
$ python3 train.py --dataset selfie2anime --cuda
```

If your CUDA memory is less than 12G, please use.
```bash
$ python3 train.py --dataset selfie2anime --cuda --light
```

If you want to load weights that you've trained before, run the following command (e.g step 10000).

```bash
$ python3 train.py --dataset selfie2anime --cuda --netG_A2B weights/selfie2anime/netG_A2B_100000.pth --netG_B2A weights/selfie2anime/netG_B2A_100000.pth --netD_A weights/selfie2anime/netD_A_100000.pth --netD_B weights/selfie2anime/netD_B_100000.pth --netL_A weights/selfie2anime/netL_A_100000.pth --netL_B weights/selfie2anime/netL_B_100000.pth
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

_Junho Kim, Minjae Kim, Hyeonwoo Kang, Kwanghee Lee_ <br>

**Abstract** <br>
We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module 
and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus 
on more important regions distinguishing between source and target domains based on the attention map obtained 
by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes 
between domains, our model can translate both images requiring holistic changes and images requiring large shape 
changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided 
model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. 
Experimental results show the superiority of the proposed method compared to the existing state-of-the-art 
models with a fixed network architecture and hyper-parameters.

[[Paper]](https://arxiv.org/pdf/1907.10830) [[Authors' Implementation (TensorFlow)]](https://github.com/taki0112/UGATIT) [[Authors' Implementation (PyTorch)]](https://github.com/znxlwm/UGATIT-pytorch) 

```
@inproceedings{
    Kim2020U-GAT-IT:,
    title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
    author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwang Hee Lee},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=BJlZ5ySKPH}
}
```