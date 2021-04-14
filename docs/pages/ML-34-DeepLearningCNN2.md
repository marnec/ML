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
Suppose we want to build a classifier that is able to recognize if an input image is the picture of a cat (<a href="#fig:cnn">Figure 86</a>). For the sake of this example we will use small input images with dimensions: $39 \times 39 \times 3$. By convolving 10 $3 \times 3 \times 3$ filters with *valid* padding and stride $s^{[1]}=1$, we will obtain an output $a^{[1]} \in 37 \times 37 \times 10$.

A second convolutional layer with 20 $3\times 3 \times 3$ filters with *valid* padding and stride $s^{[2]}=2$ will give $a^{[2]} \in 17 \times 17 \times 20$. And a further layer with 40 $3 \times 3 \times 3$ filters with *valid* padding and stride $s^{[3]}=2$ will give an output $a^{[3]} \in 7 \times 7 \times 40$.

The last output layer $a^{[3]}$ can be reshaped into a $1960 \times 1$ array and fed into a logistic and softmax activation function to obtain the prediction $\hat{y}$.


    

<figure id="fig:cnn">
    <img src="{{site.baseurl}}/pages/ML-34-DeepLearningCNN2_files/ML-34-DeepLearningCNN2_6_0.svg" alt="png">
    <figcaption>Figure 86. A simple convolutional neural network with 3 convolutional layers and one final dense layer.</figcaption>
</figure>

As shown in this example, the typical trend of a convolutional neural network is that the width and height of the activation matrices will generally shrink, while the number of channels will generally grow as we advance deeper in the network. At each convolutional step, the values $f^{[l]}, s^{[l]}, p^{[l]}$ add themselves to the list of hyperparameters that need to be set. Different values of these hyperparameters can give fundamentally different results.

## Types of layer in a convolutional network
In a typical convolutional network there are usually three types of layer:

* Convolutional layer (CONV): this is the type of layer that we just described 
* Pooling layer (POOL)
* Fully connected layer (FC)

Although it is possible to design good neural networks that just use CONV layers, most architectures will use combinations of the three.

### Pooling layers
Pooling layers are used to reduce the size of the representation, to speed up the computation as well as make the features detected more robust.
A pooling layer is a layer that performs an operation on a window of cells of dimensions $f \times f$. In **Max pooling** for example, the operation is taking the max value of the value in the $f \times f$ window (<a href="fig:maxpool">figure below</a>).


    

<figure id="fig:maxpool">
    <img src="{{site.baseurl}}/pages/ML-34-DeepLearningCNN2_files/ML-34-DeepLearningCNN2_8_0.svg" alt="png">
    <figcaption>Figure 87. Max pooling on a $4 \times 4$ input with a $2\times 2$ window with stride $s=2$</figcaption>
</figure>

The inutition behind max pooling is to "de-noise" the feature detected in the input feature, which should be represented by high values in the input, and exclude low values. Despite this the widely accepted intuition of max pooling, to the best of my knowledge I don't know if anyone 


```python
company = 'Porcia Magazzino'
people = 117
postazioni_rot = 29
postazioni_fix = 24
task = 'pianificare persone in postazioni con rotazione settimanale'



```


```python

```
