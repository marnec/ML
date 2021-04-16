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
