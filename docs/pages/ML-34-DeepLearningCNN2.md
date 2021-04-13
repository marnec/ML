---
layout: default
title: "Deep Learning - CNN - Convolutional NN"
categories: deeplearning
permalink: /ML34/
order: 34
comments: true
---

# Convolutional neural network

## Building one layer of a CNN
From what we have seen in <a href="{{site.basurl}}/ML/ML33#fig:multifilter">Figure 84</a>.


    

<figure id="fig:onelayercnn">
    <img src="{{site.baseurl}}/pages/ML-34-DeepLearningCNN2_files/ML-34-DeepLearningCNN2_2_0.svg" alt="png">
    <figcaption>Figure 85. A single layer of a convolutional neural network, with an input matrix $a^{[0]}$, convolved with two learned filters $w^{[1]}$ and the ReLU activation function applied to the two channels of the produced output images $g(w^{[1]}a^{[0]})$.</figcaption>
</figure>


In building one layer of a convolutional neural network, the input image, analogously to any neural network, can also be called $a^{[0]}$. The two filters play a similar role to the weights of the first layer of the neural network $w^{[1]}$. Each filter is multiplied and summed to the input image and produces a $4\times 4$ output, which is similar to $w^{[1]}a^{[0]}$. For each of these outputs, we can add a bias $b_i$, which is a single number $\in \mathbb{R}$ that is added to each cell of the output image. A non linearity function is applied to each output $g(w^{[1]}a^{[0]})$. Finally the two outputs are stacked together to generate the final output of one layer of a convolutional neural network $a^{[1]}$.

In the example in <a href="#fig:onelayercnn">Figure 85</a> we have two filters and this is the reason why we end up with 2 channels in the output, but if instead we had for example 10 filters, we would have a $4 \times 4 \times 10$ output. Such a layer would have $3 \times 3 \times 3 + 1= 28$ parameters plus bias, multiplied $ \times 10$ for each filter, for a total of $280$ parameters for this layer. The nice thing about convolutional neural network, is that **number of parameters is independent from the size of the input image**. Even if the input image was a 1000 x 1000 pixels image, a layer such as what we have described, would still have 280 parameters. This means that with a very small number of parameters we can add filters for detecting a number of features of input images of any dimension. This property of convolutional neural networks make them **less prone to overfitting**.

### Notation for one convolutional layer

Notation for convolutional networks can be overwhelming, so to disambiguate from now on we will use the following notation. For one convolutional layer $l$:

One filter has dimension $f^{[l]}$

A convolution can have padding $p^{[l]}$ and stride $s^{[l]}$

The input of layer $l$ is $n_H^{[l-1]} \times n_W^{[l-1]} \times n_c^{[l-1]}$, where

* $H$ and $W$ are the height and width respectively
* $[l-1]$ indicates that the input is the activation values coming from the previous layer 
* $n_c$ is the number of channels in the previous layer

The output $a^{[l]}$ is $n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$ where

* $n_H^{[l]} = \left \lfloor \frac{n_H^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1 \right \rfloor$ 
* $n_W^{[l]} = \left \lfloor \frac{n_W^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1 \right \rfloor$
* $n_c^{[l]}$ is the number of filters

Each filter has size $f^{[l]} \times f^{[l]} \times n_c^{[l-1]}$

Output for single examples $a^{[l]}_i$ are combined in a matrix $A^{[l]}$ with dimensions $m \times n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$

Weights from all filters combined have dimensions $f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}$

The bias will be a number for each filter, so a vector of $n_c^{[l]}$ elements

## Simple convolutional network

