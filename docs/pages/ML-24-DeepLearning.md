---
layout: default
title: "Deep Learning"
categories: deeplearning
permalink: /ML24/
order: 24
comments: true
---

# Deep learning
While the concept of deep learning has been around since many years, it really took off some years ago. On one hand, from the advent of computers and internet we ha started to accumulate an enormous amount of data. On the other hand computational power has increased to the point that training very large neural network is now not only possible, but relatively easy. These two fact combined are the reason behind the popularization of deep learning algorithm, since large amount of data (more precisely labeled data) fed into very large neural networks, produce results unachievable by other models.

So in one word **scale** has been driving deep learning, scale of labeled data, of computational power and of the algorithm.

Incidentally many new technologies (i.e. types of neural networks) have been invented while trying to run large algorithms faster: for example, one of the fundamental breakthrough in ML has been switching from a sigmoid function to a RELU function (<a href="#fig:sigmoidrelu">Figure 40</a>). This is due to the fact that in the regions far from $0$, the parameters change very slowly, while with the ReLU the gradient descent is much more efficient.


    

<figure id="fig:sigmoidrelu">
    <img src="{{site.baseurl}}/pages/ML-24-DeepLearning_files/ML-24-DeepLearning_2_0.png" alt="png">
    <figcaption>Figure 40. Comparison between sigmoid function and ReLU</figcaption>
</figure>

## Deep representation
Why do deep neural networks work well? Deep neural networks build a hierarchical representation of training data.

Early layers of neural networks detect simpler functions and compose them together in following layers of the neural network in more complex functions (a possible example in <a href="#fig:audioexample">Figure 41</a>).


    

<figure id="fig:audioexample">
    <img src="{{site.baseurl}}/pages/ML-24-DeepLearning_files/ML-24-DeepLearning_5_0.png" alt="png">
    <figcaption>Figure 41. An hypothesized example of increasingly complex features learned from a 5 layers-deep neural network trained on audio sources.</figcaption>
</figure>

A result from circuit theory states that:

> there are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute.
    
Let's try to illustrate this with an example: as we have seen in <a href="{{site.basurl}}/ML/ML11">ML11</a> we can calculate logical functions with neural networks. Let's say we want to calculate the combined $\text{XOR}$ of our input vector $x$.

$$y=x_1 \text{XOR} x_2 \text{XOR} x_3 \text{XOR} \dots \text{XOR} x_n$$

To compute $y$ the depth of the neural network should be on the order of $O(\log n)$. However if we are forced to compute this function with just 1 hidden layer than in order to compute $y$ we will need $\sim 2^n$ hidden units.

## Notation
Let's take the deep neural network in <a href=#fig:deepann>the figure below</a>


    

<figure id="fig:deepann">
    <img src="{{site.baseurl}}/pages/ML-24-DeepLearning_files/ML-24-DeepLearning_8_0.png" alt="png">
    <figcaption>Figure 42. A 4-layers deep neural network</figcaption>
</figure>

We say that this neural network as $L=4$ layers; input layer is included in the number of layers. Each layer has $n^{[l]}$ number of units. In this case:

$$
\begin{aligned}
& n^{[1]}=5 \qquad n^{[2]}=5 \qquad n^{[3]}=3 \qquad n^{[4]}=n^{[L]}=1 \qquad \\
& n^{[0]} = n_x=3
\end{aligned}
$$

The activations in each layer are referred to as $a^{[l]}$ and are computed as:

$$
a^{[l]} = g^{[l]}\left(z^{[l]}\right)
$$

where $z^{[l]}$ is calculated from parameters $W^{[l]}$ and bias $b^{[l]}$.

The input layer is referred to as $x = a^{[0]}$ and the output layer as $a^{[L]} = \hat{y}$

## Forward propagation
Forward propagation for a deep neural network follows the same process as for a shallow network, which is explained in detail in <a href="{{site.basurl}}/ML/ML10">ML10</a>. The process follows the general rule:

$$
\begin{aligned}
& Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]} \\
& A^{[l]}=g^{[l]}(Z^{[l]})
\end{aligned}
$$

### Matrix dimensions
#### For a single example
Matrices mentioned in forward propagation have the following dimensions for a single training example:

$$
\begin{aligned}
z^{[l]} & \in \mathbb{R}^{(n^{[l]}, 1)} \\
W^{[l]} & \in \mathbb{R}^{(n^{[l]}, n^{[l-1]})} \\
a^{[l-1]} & \in \mathbb{R}^{(n^{[l-1]}, 1)} \\
b^{[l]} & \in \mathbb{R}^{(n^{[l]}, 1)}
\end{aligned}
$$
  
### For the whole training set
In the vectorized form across $m$ training examples the matrices have the following dimensions:

$$
\begin{aligned}
Z^{[l]} & \in \mathbb{R}^{(n^{[l]}, m)} \\
W^{[l]} & \in \mathbb{R}^{(n^{[l]}, n^{[l-1]})} \\
A^{[l-1]} & \in \mathbb{R}^{(n^{[l-1]}, m)} \\
b^{[l]} & \in \mathbb{R}^{(n^{[l]}, 1)}
\end{aligned}
$$

## Back propagation
As for a shallow neural network back propagation follows the general rule:

$$
\begin{aligned}
& dZ^{[l]} = dA^{[l]} g'^{[l]} \left( Z^{[l]} \right)\\
& dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T} \\
& db^{[l]} = \frac{1}{m} \sum dZ^{[l]} \\
& dA^{[l-1]} = W^{[l]T} dZ^{[l]}
\end{aligned}
$$

Where the derivative of the final layer (if using the sigmoid  activation function) $dA^{[L]}$ is:

$$
dA^{[L]} = -\frac{y}{a}+\frac{(1-y)}{(1-a)}
$$

## Hyperparameters
Hyperparameters are settings that influence how the neural network learns the parameters $W$ and $b$. There are many hyperparameters in deep learning:
* The learning rate $\alpha$
* number of iterations
* number of hidden layers
* number of hidden units
* choice of activation functions
* other hyperparameters

There currently is no way of determining in advance optimal hyperparameters and the only way to find out what are good values for hyperparameters is to empirically test them. For this reason, especially nowadays, training a machine learning algorithm is an highly iterative process, where the best performing model is chosen among many others.

## Testing a model
In order to select the best performing algorithm it becomes essential to have a framework to test machine learning models. The common system is to split the dataset of examples in 3 separate subsets:

* **training set**: the set of examples on which the parameters are fitted
* **cross-validation set**: (also called hold-out or development set): the set of examples of which different models are tested and hyperparameters are chosen
* **test set**: the set of examples on which the performance of a model is measured

In early days of machine learning it was widely considered a good practice to split your dataset with a 70/30% or 60/20/20% ratio. And this is fine when the number of examples are in the order of $10^{2-4}$. But for deep-learning problems, where usually the number of examples is in the order of $10^{6}$, the fraction of data in the test and dev sets is much much smaller and a good split would be a 98/1/1%.

### Mismatched train/test distributions
When dealing with large amount of data it may happen that data in the training set and data in the dev or test set come from different distributions.

For example the training set might be fed by automatic data crawlers while the dev or test set might be fed data from users. This is generally fine as long as the dev and test sets come from the same distribution. 

### Omitting the test set
Since the goal of the test set is to provide a platform to obtain an un-biased estimate of the performance of the model, it is not required to have one in case we don't need to estimate such performance. The dev set on the other hand is absolutely necessary to chose the best performing model
