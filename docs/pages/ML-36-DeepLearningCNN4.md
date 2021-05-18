---
layout: default
title: "Deep Learning - CNN - Using pre-designed networks"
categories: deeplearning
permalink: /ML36/
order: 36
comments: true
---

# Using open-source implementations 
In the previous article, we have seen many effective architectures of CNN. Many of these networks are difficult to replicate even for an intermediate practitioner starting from the research paper, due to many little details (e.g. hyperparameters), that can produce great variations in the output performance.

Luckily, many deep-learning researcher routinely publish their networks as open-source software. So, the first step to replicate an architecture from a research paper is to look online for an open-source implementation from the authors themselves.

## Transfer learning on pre-trained CNNs
When building a computer vision application rather than training the weights from scratch you can download networks with pre-trained weights and transfer this learning to a new task. The computer vision community has published many datasets upon which many models have been trained. This models have been trained for a long time, maybe weeks, employing gpus and the long process of hyperparameters exploration. Downloading a pre-trained netwrok can save months of work and function as a very good parameter initialization.

Suppose we are building an object detector that can detect 3 classes, A, B or neither. A typical pre-trained network, maybe trained on [image-net](http://www.image-net.org/), has a softmax regression layer as its final layer. In order to transfer-learning we first get rid of the softmax layer and build our softmax layer of interest, with 3 classes. The remaining layers in the network are **frozen** and will not be trained. These frozen layers will maintain their pre-learned parameters and hopefully they encode for low-level features that are general and useful for your task.

Since the early layers are frozen, you can pre-compute the final activation values of the last layer of the frozen network for all pictures of your dataset and save these activation values vectors to disk. Since these activation values vectors are the input of your shallow softmax network, you can just feed these pre-computed intermediate values to save time in the computation while experimenting with different configurations of your shallow softmax network.

The number of layers to freeze is, not strictly speaking, inversely proportional to the amount of training data for your task of interest. If a large dataset is available, you can freeze less layers, from deeper to shallower. And if your dataset is large enough you can re-train the whole network, with the advantage of having all the hyper-parameters pre-selected.

## Data augmentation
All computer vision tasks will benefit from getting more data. This is true whether you are using transfer learning or training a model from scratch. Data augmentation can be used to artificially increase the amount of data available.

A common augmentation method is **mirroring along the vertical axis**. For most tasks, an horizontal flip will not change label associated to an input. For example, flipping a picture of a cat horizontally, still shows a cat. Another common method is **random cropping**, where random crops of the original images are used as different input images. Random cropping is not a perfect method, since a crop of an image might produce a non relevant picture and actually introduce an incorrectly labeled image. These two methods are frequently used and in principle you could use **rotation**, **shearing**, **local warping** of the image and other methods, but these latter methods are less used.

Another augmentation method is **color shifting**, where each channel is shifted up or down by some constant, for example making the picture more bluish or less red. Color shifting makes learning more robust to variations of light environment and capture device. The choice of how to shift the RGB channels is based on PCA (<a href="{{site.basurl}}/ML/ML21ML21">ML21</a>) and it is sometimes called **PCA color augmentation** (from the [AlexNet article](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)).

Usually data augmentation and training is parallelized at run-time. Data augmentation is performed on one or multiple images on one process, giving a mini-batch that is passed to another process for training.

Finally data augmentation, similarly to other aspects of deep-learning has many hyperparameters, as for example how much to shift the colors or how much rotation is applied, and similarly to other aspects a good place to start with these hyperparameters are open-source implementations at a high-level and research articles at a lower level.

## Tips for doing well on benchmarks/winning competitions
There is a series of techniques that are not necessarily useful when shipping a DL algorithm for a client but that are usually employed to increase performance on a benchmark dataset (usually necessary for publication) or on a competition.

* **Ensembling**: train several networks independently and average their outputs  $\hat{y}$. Typically this increase performance up to 1-2%, which can make a difference in a competition, but will also multiply the time required for each prediction.

* **Multi-crop at test time**: run a classifier on multiple versions of test images and average their results. A typical mulit-crop technique is the 10-crop where a central crop and 4 corner crops are taken for an image and its horizontal mirror.
