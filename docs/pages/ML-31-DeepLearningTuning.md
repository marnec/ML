---
layout: default
title: "Deep Learning - Speed up learning - Tuning"
categories: deeplearning
permalink: /ML31/
order: 31
comments: true
---

# Hyperparameter tuning

## Learning rate decay
Learning rate decay is a technique where you slowly reduce the learning rate over the training iterations.

The intuition behind learning rate decay is represented in <a href="#fig:lrdecay">Figure 59</a>. When approaching the optimum during gradient descent, if the learning rate remains constant, it may diverge from the optimum. Instead, we want gradient descent to take larger steps when we are far from the optimum and smaller steps when we are close to the optimum, so that even if the model never converges, it can hover close enough to the optimum to give good results.


    

<figure id="fig:lrdecay">
    <img src="{{site.baseurl}}/pages/ML-31-DeepLearningTuning_files/ML-31-DeepLearningTuning_2_0.png" alt="png">
    <figcaption>Figure 59. A bidimensional feature space with contours of iso-values of the cost $J$ with gradient descent steps taken with constant learning rate (blue) and decaying learning rate (orange).</figcaption>
</figure>

In learning rate decay, our learning rate $\alpha$ becomes smaller each epoch, according to a system. There are different systems, among the most common way of computing $\alpha$ we have

$$
\begin{aligned}
& \alpha = \frac{1}{1+d \cdot \tau} \alpha_0 \\
& \\
& \alpha = 0.95^\tau \cdot \alpha_0 \\
& \\
& \alpha = \frac{k}{\sqrt{\tau}} \cdot \alpha_0 \\
& \\
& \alpha = \frac{k}{\sqrt{t}} \cdot \alpha_0 \\
\end{aligned}
$$

where $k$ is a constant, $t$ is the mini-batch number and $\tau$ is the epoch-number.

Sometimes instead **manual decay** is applied. This means that while the algorithm is training, you can evaluate the the learning rate needs to be tuned down and setting it manually.

## Tuning process
How to organize your hyperparameters tuning process. In deep-learning you come across many hyper-parameters: We have seen:

* the learning rate $\alpha$ (if you are using a constant learning rate)
* the momentum parameter $\beta$
* the parameters $\beta_1, \beta_2, \epsilon$ if using ADAM optimization
* the number of hidden layers
* the number of hidden units in each layer
* the number of epochs
* the learning rate decay system and its parameter and possibly its parameter $k$
* the mini-batch size $t$

In many cases, if sorted by their importance the list would be: 

1. The single most important hyperparameter in almost all situations is the learning rate $\alpha$. 
2. Second in importance come
    * the momentum parameter $\beta$, for which $0.9$ is found to be be a good default parameter. 
    * the the mini-batch size
    * the number of hidden units in layers
3. Third in importance come 
    * the number of layers (that can sometime make a huge difference)
    * the learning rate decay
4. When using an ADAM optimization algorithm usually its parameters are never tuned and the default are kept ($0.9, 0.999, 10^{-8}$)

However it is very difficult to give a general rule for the importance of hyperparameters and each model tend to behave differently.

### Hyperparameter exploration
In early days of machine learning, practitioners would sample the space of hyperparameters systematically, by testing combinations of intervals of hyperparameters (<a href="#fig:hypertune">Figure 60</a>, panel B), since it is almost impossible to know in advance which hyperparameter will have more impact on the model, but at the same time some hyperparameters tend to be much more important than others


    

<figure id="fig:hypertune">
    <img src="{{site.baseurl}}/pages/ML-31-DeepLearningTuning_files/ML-31-DeepLearningTuning_6_0.png" alt="png">
    <figcaption>Figure 60. Hyperparameter space sampling in early days of machine learning (A) and in modern days of deep-learning (B)</figcaption>
</figure>

This is done because when sampling the hyperparameter space as in <a href="#fig:hypertune">Figure 60</a> panel B, is a different set of Hyperparameters 1 and 2.


```python

```
