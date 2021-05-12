---
layout: default
title: "Deep Learning - CNN - Case Studies"
categories: deeplearning
permalink: /ML35/
order: 35
comments: true
---

# Case Studies
The past few year of research on convolutional networks applied to computer vision has focused on the design of architectures of convolutional networks. Looking at some examples of CNN designs is useful for two main reasons: first, a number of problems have already been faced by someone else, and looking at how they resolved the problem is sometimes very insightful. Second, to gain intuitions on how to design your own network it is important to look at how others have done it before. Furthermore, a lot of these idea that were specifically designed for computer vision, are cross-contaminating with other fields. We will see:

* Some classic networks, some of these works laid the foundations of modern computer vision ML:
    * LeNet-5
    * AelxNet
    * VGG

* ResNet, or **residual network**, an example of building a very deep (152) layers neural network effectively. 
* A case study of **inception** neural network
* MobileNets for low-computation environments

## Classic networks
### LeNet-5
The goal of the network was to recognize handwritten digits, its architecture is shown in <a href="#fig:lenet5">Figure 90</a>. LeNet-5 architecture starts with a $32 \times 32 \times 1$ grayscale image. In the first step a set of 6 $5 \times 5$ filters with stride $s=1$ and valid padding. This reduce the image to an $28 \times 28 \times 6$ output representation in CONV1 layer. Then the LeNet-5 applies average pooling with a $2 \times 2$ filter with stride $s=2$ that reduces the image to $14 \times 14 \times 6$. Right now a max-pooling is usually used but at the time, average pooling was more popular. The CONV2 layer is obtained by applying 16 $5 \times 5$  filters with stride $s=1$ and valid padding, which result in a $10 \times 10 \times 16$ output representation. When the LeNet-5 network was published padding was not used and in fact it is not applied to the CONV1 and CONV2 layers of this network. A second average pooling layer POOL2 is obtained by a $2 \times 2$ filter with stride $s=2$, which results in a $5\times 5 \times 16$ representation. This 3D matrix is reshaped in a $400 \times 1$ vector in the FC1 layer which is fully connected two FC2 ($120 \times 1$) and FC3 ($84 \times 1$). Finally a $10 \times 1$ output layer allow the multi-class classification. In modern days networks it would be a softmax classifier while in the LeNet-5 a less used classifier was employed.


    

<figure id="fig:lenet5">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_2_0.svg" alt="png">
    <figcaption>Figure 90. Architecture of the LeNet-5 classic network</figcaption>
</figure>

The LeNet-5 network is small by today's standard with approximately 60.000 thousand parameters in total, where nowadays we see networks in the range of 10-100 million parameters. Some other differences in the LeNet-5 architecture from modern standard in CNNs and neural networks in general is the activation function, where the sigmoid and tanh were used back then while we now almost always use ReLU. Furthermore the activation function was applied after pooling, while we now usually apply it before pooling. A couple of things are still designed in the same way in modern networks: as you go deeper in the network, there is a gradual shrink in height and width and the growth in the number of channels; convolutional and pooling layers alternate, even if not always with a 1:1 ratio. 

### AlexNet
The AlexNet has a similar architecture to LeNet-5 (<a href="#fig:alexnet">Figure 91</a>), in the sense that it alternates pooling layers to convolutional layers with the latest layers being fully connected. It is also similar in the fact that the number of channels grows further down in the network. However it has also many differences. AlexNet uses ReLU activation functions and makes use of *same* padding to prevent excessive shrinking in the width and height dimension. AlexNet has 1000 output classes assigned with a final softmax classifier layer. It has 60 million total parameters and this, together with the possibility of being trained on much more data were the reasons behind its remarkable performance.


    

<figure id="fig:alexnet">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_5_0.svg" alt="png">
    <figcaption>Figure 91. The AlexNet architecture</figcaption>
</figure>

The AlexNet has a fairly complicated architecture with many hyperparameters. This contrasts with the next classic network, the VGG-16

### VGG-16
A remarkable difference in the design of the VGG-16 network is that, compared to other networks it has a relatively simple architecture (<a href="#fig:vgg16">Figure 92</a>). When designing the VGG-16 network, the decision was taken to only employ convolutional layers with $f=3,s=1$ and *same* padding, and max-pooling layers with $f=2,s=2$. This really simplify the network architecture. The VGG-16 is deeper than the LeNet-5 and the AlexNet; it has 16 layers with parameters and a $\approx 138 M$ parameters in total, which makes it a large network even for today's standards. The VGG-16 architecture alternates 2 or 3 convolutional layers to a max-pooling layer, gradually increasing the number of channels and decreasing in height and width the representation. The number of channels increases in powers of 2, from 64 to 128, to 256 and finally to 512. Another even deeper version exists, the VGG-19 with 19 layer with parameters, but since it performs in most cases as the VGG-16, the latter is preferred since it has fewer parameters and it is thus faster to train. 


    

<figure id="fig:vgg16">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_7_0.svg" alt="png">
    <figcaption>Figure 92. Architecture of the VGG-16 network, representations are only shown as their dimensions since the aspect of most of them would make the figure unreadable.</figcaption>
</figure>

## ResNets
Very deep neural networks are difficult to train because they tend to suffer from vanishing or exploding gradients (<a href="{{site.basurl}}/ML/ML28">ML28</a>). With **skip connection**, units in a layers are connected directly to units in much deeper (or shallower) layers skipping all intermediate layers. **Residual networks** (ResNets) are built with skip connections and they allow to train very deep neural networks with up to hundred of layers).

ResNets are built off of **residual blocks**. In a normal neural network for information to flow from an activation unit $a^{[l]}$ to $a^{[l+2]}$ it would need to undergo two linear and two non-linear transformations (<a href="#fig:resblock">Figure 93</a>), in a residual block $a^{[l]}$ takes a short cut and it is plainly added to $z^{[l+2]}$ before applying the second non-linearity function, so that $a^{[l+2]}=g(z^{[l+2]}+a^{[l]})$


    

<figure id="fig:resblock">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_9_0.svg" alt="png">
    <figcaption>Figure 93. Concept of skip connection in a residual block. The flow of information in a normal neural network and the shortcut took by a residual block</figcaption>
</figure>

What the inventors of residual networks proved, was that by stacking many residual blocks into a residual network, it was possible to train much deeper networks compared to "plain networks" as they refer to into their manuscript. While training a deep neural network, as you increase the number of layers the error on the training set tends to decrease until a certain point. After a certain number of layers the training error tends to go back up, where in theory the deeper the network, the better the training error. In practice, due to vanishing or exploding gradient (<a href="{{site.basurl}}/ML/ML28">ML28</a>, panel A). With a Resnet, even if number of layers get deeper the training error keeps getting lower as we expect, even with networks 100 layers deep. Recent research employed residual networks 1000 layers deep not suffering from performance deterioration despite the large number of layers. 


    

<figure id="fig:resnetperf">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_11_0.svg" alt="png">
    <figcaption>Figure 94. Error on the training set as a function of the number of layers in a plain network (A) and in a residual network (B)</figcaption>
</figure>

The reason why ResNet allow very deep networks to not loose performance stands in their ability to easily represent the identity function, rendering the network simple when the data requires it. Let's see what that means: when $a^{[l]}$ skips the connection it is injected in the computation of $a^{[l+2]}$:

$$
\begin{split}
a^{[l+2]} & = g \left( z^{[l+2]} + a^{[l]}\right) \\
&=g(W^{[l+2]} a^{[l+1]} + b^{[l+2]} + a^{[l]})
\end{split}
$$

If we are using $L_2$ regularization weight decay that would tend to shrink the value of $W^{[l+2]}$ and, less importantly, of $b^{[l+2]}$. If both those elements tend to zero, then

$$
a^{[l+2]} = g (a^{[l]})
$$

Since we are using ReLU activation function, then $g (a^{[l]}) = a^{[l]}$, which means

$$
a^{[l+2]} = a^{[l]}
$$

This means that the identity function is easy for a residual block to learn. This, in turn means that having the two layers between $a^{[l]}$ and $a^{[l+2]}$ doesn't hurt performance, because when the data requires it, it will be easy for the network to just ignore those two layers. On the contrary, if the data is complex and requires a more complex representation, those layers can learn different and relevant parameters for the output representation. In fact, what goes wrong in very deep plain networks is that in deeper layers it becomes increasingly different to learn even identity functions and so the performance will decay if the network is too complex compared to the target function.

### ResNet requires dimension uniformity
One requirement of residual blocks is for $z^{[l+2]}$ and $a^{[l]}$ to have the same dimensions since they are added together. In order to achieve dimension uniformity usually *same* padding is used between convolutional layers. Pooling layers are positioned after the first, eight, sixteenth and twenty-eighth convolution. Each time a pooling layer is applied an adjustment to representation dimensions is required, in order to achieve the required dimension uniformity. To do that, a parameter matrix $W_s$ is multiplied to $a^{[l]}$. If $a^{[l+2]}$ has a side of $256$ and $a^{[l}$ a side of $128$, than $W_s \in \mathbb{R}^{256\times 128}$ so that $a^{[l]}$ will have side $256$. the parameter matrix $W_s$ could either be a matrix of learned parameters or a fixed matrix that just implements zero padding.


    

<figure id="fig:resarchitecture">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_13_0.svg" alt="png">
    <figcaption>Figure 95. A plain CNN and a ResNet of 34 layers. All $3 \times 3$ convolutions have *same* padding so that dimensionality is uniform between $z^{[l+2]}$ and $a^{[l]}$. Pooling layers are positioned after the first, eight, sixteenth and twenty-eighth convolution. Pooling layers have the dimensions of the representation and are thus shown as $/2$ in the label of the convolutional layer to which they are applied.</figcaption>
</figure>

## Inception network
When designing CNN architecture, an idea that opens a new range of possibilities, is using a $1 \times 1$ convolution. It could seem that such a convolution would be like multiplying a matrix by a number, but the effect of $1 \times 1$ convolution are quite different.

### Network in network 
A $1 \times 1$ filter produces a representation with the same width and height of the input image, but the number of channels of the output representation is equal to the number of filters convolved. So, when convolving a $6 \times 6 \times 8$ input with one $1 \times 1$ filter, we will obtain a $6 \times 6 \times 1$ output representation. This means that when dealing with an input with more than one channel, all values in the channels at coordinates $i,j$ are linearly combined in a new value. When convolving a $6 \times 6 \times 8$ input with two $1 \times 1$ filters, we obtain a $6 \times 6 \times 2$ output representation (<a href="#fig:metanet">Figure 96</a>). 

Imagine that the convolution of the two $1 \times 1$ filters happens is performed in parallel. At each step of the convolution, each filter is moved to a new position and two values (one for each filter) are computed by linearly combining all values in the channels of the input representation with the single value of each filter. This operation can be represented as a fully connected network, with 8 input units (more generally, $n_c$ input units) and 2 output units (more generally, as many output units as the number of filters). This **inner network**, or **network in network**, outputs the values for the channels of a single position of the output representation and these outputs depend on a parameter matrix $W_\text{inner} \in \mathbb{R}^{ \text{#filters} \times n_c}$  


    

<figure id="fig:metanet">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_16_0.svg" alt="png">
    <figcaption>Figure 96. Network in network concept represented for a single step of the convolution of a $6 \times 6 \times 8$ input with a pair of $1 \times 1$ filters, producing an network in the network, a fully connected layer with parameters $W \in \mathbb{R}^{2\times 8}$. The value of each output unit of this inner network is one position of one of the channels of the $6 \times 6 \times 2$ output representation</figcaption>
</figure>

While the architecture of the network in network described in [this article](https://arxiv.org/abs/1312.4400) is not widely used, the idea of network in network as being inspirational for many works, including inception network. 

$1 \times 1$ convolutions can be used to reduce the number of channels and speed up computation: when going deeper in a convolutional network the width and height tends to shrink due to the usage of *valid* padding or of pooling layers, while the number of channels tends to grow. A number of filters (smaller than the input $n_c$) can be used to reduce the number of channels while retaining the width and height of the output representation.

Alternatively, the number of $1 \times 1$ filters applied can leave $n_c$ unvaried while adding non-linearity and thus allowing to learn more complex functions.

## Inception networks
When designing a layer for a CNN you have to take decision on how to structure the architecture. You might have to pick a filter with specific dimensions or maybe a pooling layer. The inception network removes the need to choosing between these different options and do all at the same time.

The concept behind the inception network is to apply all relevant options to an input and let the optimization process decide what combination of filters or pooling better fits the data. 

Suppose we have a $28 \times 28 \times 192$ input. Instead of choosing the design of the layer that takes this input, combines multiple options in the same output representation by stacking them in different channel blocks. For example we could apply 64 $1 \times 1$ filters, which output a $28 \times  28 \times 64$ volume, 128  $ 3 \times 3 $ filters with *same* padding, which outputs a $28 \times 28 \times 128$ volume, 64  $ 5 \times 5 $ filters with *same* padding, which outputs a $28 \times 28 \times 64$ volume, and max-pooling which outputs a $28 \times 28 \times 32$ volume, granted that a *same* padding and a stride $s=1$ are used (<a href="#fig:inclayer">Figure 97</a>)


    

<figure id="fig:inclayer">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_19_0.svg" alt="png">
    <figcaption>Figure 97. An inception layer. An input representation with 192 channels is subject to 3 filters and a pooling; the results of all the operations are stacked in a single output representation</figcaption>
</figure>

### Computational cost of an inception layer
The inception layer as formulated above has a problem of computational cost: just the cost of the volume resulting by the convolution of the $5 \times 5$ filters is to compute $120 \cdot 10^6$ multiplications. In fact, each cell of the $28 \times 28 \times 32$ output volume is computed by computing the convolution a $5 \times 5 \times 192$ filter. Luckily the concept of $1 \times 1$ convolution helps reduce the necessary computations by a factor of 10.

Instead of convolving the $5 \times 5$ filter directly, we could reduce the input volume (and consequently the computational cost of the operation) to $28 \times 28 \times 16$ by applying 16 $1 \times 1$ filters. This intermediate reduced volume, sometimes called the **bottleneck layer** (<a href="#fig:bottleneck">Figure 98</a>), can be then used as input for the original series of 32 $5 \times 5$ filters, which still produces a $28 \times 28 \times 32$ volume. The input and output dimensions have remained unchanged but the computational cost has drastically reduced: we now have to perform $[28 \cdot 28 \cdot 16 \cdot (1 \cdot 1) 192] + [28 \cdot 28 \cdot 32 \cdot 5 \cdot 5 \cdot 16]=12.4 \cdot 10^6$.


    

<figure id="fig:bottleneck">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_21_0.svg" alt="png">
    <figcaption>Figure 98. A bottleneck layer, a $1 \times 1$ convolution used to reduce the computational cost of a larger convolution</figcaption>
</figure>

It turns out that by implementing a bottleneck layer you can shrink down the representation significantly without apparently compromising the performance of the neural network.

### Building an inception network
The inception module takes as input the representation volume from a previous layer. Building upon the inception layer example used up until this point, to build a complete inception module a bottleneck layer is applied before the $3 \times 3$ and $5 \times 5$ convolutions to reduce the computational cost of the operation. The 64 $1 \times 1$ filters that produce the 64 channels in the final output don't require a bottleneck layer (they already are $1 \times 1$ filters). Finally, max-pooling is applied directly to the input volume, with the unusual configuration of *same* padding and stride $s=1$. This max-pooling layer produces a $28 \times 28 \times 192$ volume (where 192 is the $n_c$ of the input volume), that is shrunk to $28 \times 28 \times 32$ by applying 32 $1 \times 1$ filters. Finally all intermediates output volumes are concatenated together in a unique volume, which is the output of the inception module (<a href="#fig:incmod">Figure 99</a>) 


    

<figure id="fig:incmod">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_23_0.svg" alt="png">
    <figcaption>Figure 99. An inception module A more detailed description of the inception module architecture for the inception layer in <a href="#fig:inclayer">Figure 97</a>. Bottleneck layers are shown in white. The $1 \times 1$ convolution applied after the max-pooling layer, is used to shrink its channels, which would be equal to the $n_c$ of the input layer (192), but are reduced to 32 in the final output volume.</figcaption>
</figure>

multiple inception modules are chained together to build an inception network, where the output of a module is the input of the following module. In a full inception network sometimes, some additional pooling layers are placed immediately after an input of an inception module. The final layer of an inception network is usually a softmax layer. However there may be other softmax layers along a full inception network that branch out of the flow of the network to produce intermediate predictions. This is done to check if predictions produced by a smaller network are good enough or even better than those produced by the full network. In fact, branching out softmax layers can have a regularizing effect on predictions, since it produces output from smaller, less complex (sub-)networks. 

The inception network has been first proposed by Google with the name of GoogLeNet (in honor of the LeNet-5 network) in this [research article](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf) from 2015. From its publication newer versions of the inception network were published and (at least)  one of these versions combine the inception network and the ResNet, implementing skip-connections in the inception module.

## MobileNets
MobileNet are andother foundational convolutional neural network architecture. Using MobileNets allows to built and deploy netwroks in low compute environments, such as mobile phones. In fact, most of the architectures seen until now are very computationally expensive. The key idea behind MobileNts is that of Normal vs **depthwise separable convolutions**.

### Depthwise separable convolution
In a normal convolution we have an input image with dimensions $n \times n \times n_c$, where $n_c$ is the number of channels, which is convolved with a filter of dimensions $f \times f \times n_c$. Given a $6 \times 6 \times 3$ image and a $3 \times 3 \times 3$ filter, to convolve them the filter is slid in all 16 possible positions to produce a $4 \times 4$ output (<a href="{{site.basurl}}/ML/ML33#fig:volconv">Figure 82</a>). If we have $n_c'=5$ filters, our output will be $4 \times 4 \times 5$. The total number of computations necessary for this convolution is #filter parameters times #filter positions times #filters: $(3 \cdot 3 \cdot 3) \cdot (4 \cdot 4) \cdot 5 = 2160$.


    

<figure id="fig:depthconv">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_26_0.svg" alt="png">
    <figcaption>Figure 100. A classic convolution of a $3 \times 3 \times 3$ filter over a $6 \times 6 \times 3$ input image.</figcaption>
</figure>

In contrast to the normal convolution, the depth-wise separable convolution works in two steps: a depthwise convolution followed by a pointwise convolution

* Depth-wise convolution starts with a $n \times n \times n_c$ input as in a normal convolution. However, instead of having a number of $f \times f \times n_c$ filters, we have $n_c$ filters of dimensions $f \times f \times 1$. Each filter is applied to only one input channel. In <a href="#fig:depthconv">Figure 100</a> we have a $6 \times 6 \times 3$ input image and three $3 \times 3$ filters. Filter color reflects on which input channel they are applied. This approach produce $n_\text{out} \times n_\text{out} \times n_c$ output, in this example a $4 \times 4 \times 3$ volume. The number of computations required for this step are: #filter params times #filter positions times #filter. In this example $(3 \cdot 3) \cdot (4 \cdot 4) \cdot 3=432$

* Point-wise convolution starts with the output of depth-wise convolution with dimensions $n_\text{out} \times n_\text{out} \times n_c$ ($4 \times 4 \times 3$ volume in the example) and convolve it with $n_c'$ filters with dimensions a $1 \times 1 \times n_c$. This produces a $n_\text{out} \times n_\text{out} \times n_c'$ volume, which is the exact result of normal convolution. The computational cost of pointwise convolution is #filter params times #filter positions #filters; in the example $(1 \cdot 1 \cdot 3) \cdot (4 \cdot 4) \cdot 5=240$

So, for our example, in the case of a normal convolution the cost is $2160$ computations while in the case of a depthwise separable convolution the cost is  $432 + 240 = 672$ computations, in other words 31% of the computations required by a normal convolution. In general the ratio of the computational cost of depthwise separable convolution to normal convolution is given by

$$
\frac{1}{n_c'}+\frac{1}{f^2}
$$

which for a more typical $n_c'=512$ and $f=3$ would give a depthwise separable convolution needing 10% of the computations needed by a normal convolution.

### MobileNet architecture
The idea of MobileNet is to replace any normal convolutional operation with a depthwise separable convolutional operation. The MobileNet v1 has a specific architecture where 13 blocks of depthwise separable convolution operations were chained from the input, followed by a pooling layer, a fully connected layer and a softmax classifier (<a href="#fig:mobilenet">Figure 101</a>, top panel). 

MobileNet v2 ([Sandler et al. 2019](https://arxiv.org/abs/1801.04381)) is an improvement over the original MobileNet (<a href="#fig:mobilenet">Figure 101</a>, bottom panel). MobileNet v2 introduces two main changes in the depthwise separable convolution block:

* the addition of a **residual connection** just like the residual (or skip) connection of a ResNet.
* It adds an **expansion layer** before the depthwise and pointwise (here called **projection**) convolutional layers


In [Sandler et al. 2019](https://arxiv.org/abs/1801.04381), the updated depthwise separable convolution block of MobileNet v2, which is called **bottleneck block**, is applied 17 times and is followed by a pooling layer, fully connected layers and a final softmax classifier. 


    

<figure id="fig:mobilenet">
    <img src="{{site.baseurl}}/pages/ML-35-DeepLearningCNN3_files/ML-35-DeepLearningCNN3_29_0.svg" alt="png">
    <figcaption>Figure 101. MobileNet v1 (top) and v2 (bottom) architectures</figcaption>
</figure>

Given an input with dimension $n \times n \times n_c$, the MobileNet v2 bottleneck will pass the input directly to the output by the skip connection; in the main non residual branch of the computation, it will apply an **expansion operator**, a fairly large number ($e$) of $1 \times 1 \times n_c$ dimensional filters. A factor of expansion of 6 is quite typical in MobileNet implementations so, given a $n \times n \times 3$ input we would have a $n \times n \times 18$ volume after expansion. After this step the bottleneck block applies the two steps of a depthwise separable convolution: a depthwise convolution which gives an $n \times n \times n_\text{e}$ and a pointwise convolution which returns a $n \times n \times n_c'$ volume. Since the number of $1 \times 1$ pointwise filters ($n_c'$) is smaller than then number of $1 \times 1 \times n_c$ expansion filters ($e$), this step is called **porjection**, because it projects $e$ down to $n_c'$.

The bottleneck block accomplishes two things:

* With the expansion step it increases the size of the representation enabling the neural network to learn a more complex function
* The pointwise convolution projects the representation down to a smaller set of values, reducing the memory requirement of the model
