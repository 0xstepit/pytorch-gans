# GENERATIVE ADVERSARIAL NETWORKS

<p align="center">
  <img src="/images/gan.png" width="250">
<p/>

This repo contains my implementation of several GANs models. All the models are generated within Jupyter notebooks and in a tutorial fashion. I tried to include the most important concepts related to the theory and I tried to make the codes clearer possible. All the functions and classes are commented, and when useful I also included pictures in order to better explain the concepts.

# Table of Contents

* [Introduction to GANs](#introduction-to-gans)
* [Usage](#usage)
* [Content](#content)
* [Implementations](#implementations)
  - [Vanilla GANs](#vanilla-gans)
  - [Deep Convolutional GANs](#deep-convolutional-gans)

## Introduction to GANs
GANs are a particular __generative model__ crafted by Ian Goodfellow et al. in 2014 with the aim of estimating the distribution behind data. Before diving into the models, it is better to give a context for them! Let's say we have a data set that comes from an unknown probability distribution \(P_{data}\). Our goal is to find a distribution <img src="https://render.githubusercontent.com/render/math?math=P_{\theta}">, with parameters \(\theta\), that approximates well \(P_{data}\). Instead of trying to directly identify \(P_{\theta}\), for example with __Maximum Likelihood Estimation__ (MLE), we can make a __Neural Network__ (NN) learn a map from a known distribution, i.e. Gaussian, to the desired distribution. In this case \(\theta\) contains the parameter of the network. This approach is motivated by the fact that it allows workig with distributions defined in a low dimension manifold (don't be scared, this concept will be explained in the Notebook 3 :wink:) and sometimes is more useful the possibility to easily generate samples than the explicit knowledge of the density distribution.

The vanilla implementation of GANs, is composed by two distinct Multilayers Perceptron architectures in which a network is trained to generate fake data, the __Generator__, and the second network is trained to discriminate between real and fake data, the __Discriminator__. If we are lucky, at the end of the training, the Generator will have learned to generate samples that resemble those coming from \(P_{data}\).

<img src="/images/simple-gan.png" width="700">

## Usage

Since this repo contains only Notebooks, their usage is very simple. My suggestion is to run the Notebook in a Colab environment so that you can exploit hardware accelerators as GPU. To do so open a terminal and download the repo:

`$ git clone https://github.com/stepyt/PyTorch-GANs`

To open notebooks in a local __Jupuyter__ environment, in the terminal:

`$ jupyter-notebook &`
`$ cd PyTorch-GANs/notebook`

and then open the tutorial you wish. To open notebooks in a Colab environment please navigate to your personal __Google Drive__. If it is not already present create a folder named `Colab Notebooks`. Inside this folder creates another folder with the same name of the repo `PyTorch-GANs`, and load into it the folder `notebooks` downloaded with the repo. Now you can easily open inside Colab any notebook.

## Content

Brief description of the __folders__:
1. `/data`: contains the data sets used inside the notebooks;
2. `/images`: contains images included in the notebooks and in this READ_ME;
   * `/images/models`: contains the images saved during the models training;
3. `/notebooks`: contains the notebooks;
4. `/weights`: contains the weights saved at the end of the training. In this way you can just load the weights inside the models and play with them

## Implementations

### Vanilla GANs [[Notebook]](notebooks/Vanilla-GANs.ipynb)
Contains my implementation of the paper ["Generative Adversarial Network" (2014)](https://arxiv.org/abs/1406.2661). This is a very easy implementation, all the steps are well explained and easy to understand. This is perfect for a first approach to GANs :rocket:. At the end of this tutorial you will be able to understand the basic concepts of GANs, their architecture and how to train them 🧠:bomb:. You will train the model on MNIST and your output will be something like:

<p align="center">
  <img src="/images/models/gan/gan_gif.gif" width="200">
</p>

### Deep Convolutional GANs [[Notebook]](notebooks/DCGANs.ipynb)
Contains my implementation of the paper ["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2016)](https://arxiv.org/abs/1511.06434). The model is an extension of the Vanilla one, in which convolutional and transposed convolutional layers are used. The data set used is the CelebA, and the result is an ugly-face generator. You will explore how to include transposed and standard convolution to obtain something like that:

<p align="center">
  <img src="/images/models/dcgan/dcgan_gif.gif" width="200">
</p>

In this notebook you will explore also how to walk inside the latent space of our Generator to interpolate samples:

<p align="center">
  <img src="/images/models/dcgan/interpolation.png" width="300">
</p>

The notebook ends with a simple example on how to save and load the weight of your trained model.
