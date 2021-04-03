---
layout: default
title: "Deep Learning - Speed up learning - Tuning"
categories: deeplearning
permalink: /ML31/
order: 31
comments: true
---

# Hyperparameter tuning

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

### Approaches to hyperparameter tuning
When approaching hyperparameter tuning there two choices, which are based on the amount of computational power available and the complexity (in terms of size of the network and training data) of the model: 

* If computational resources are scarce relatively to the complexity of the model, then one tends to babysit one model. A single model is trained and its hyperparameters are tuned day by day based on the progress of the training (<a href="#fig:tuneapproach">Figure 60</a>, panel A).

* If computational resources are abundant, it is always better to train many models in parallel, each with a different set of hyperparameters (<a href="#fig:tuneapproach">Figure 60</a>, panel B).


    

<figure id="fig:tuneapproach">
    <img src="{{site.baseurl}}/pages/ML-31-DeepLearningTuning_files/ML-31-DeepLearningTuning_4_0.png" alt="png">
    <figcaption>Figure 60. Approaches to hyperparameter tuning</figcaption>
</figure>

### Hyperparameter exploration
In early days of machine learning, practitioners would sample the space of hyperparameters systematically, by testing combinations of intervals of hyperparameters (<a href="#fig:hypertune">Figure 61</a>, panel B), since it is almost impossible to know in advance which hyperparameter will have more impact on the model, but at the same time some hyperparameters tend to be much more important than others


    

<figure id="fig:hypertune">
    <img src="{{site.baseurl}}/pages/ML-31-DeepLearningTuning_files/ML-31-DeepLearningTuning_6_0.png" alt="png">
    <figcaption>Figure 61. Hyperparameter space sampling in early days of machine learning (A) and in modern days of deep-learning (B)</figcaption>
</figure>

This is done because when sampling the hyperparameter space as in <a href="#fig:hypertune">Figure 61</a> panel B, is a different set of Hyperparameters 1 and 2.

### Scale of hyperparameters
Choosing hyperparameters at random doesn't mean that they all need to be randomly picked from a uniform distribution, each parameter should have the appropriate scale (<a href="#fig:scales">Figure 62</a>). The number of hidden layers $n^{[l]}$ can be sampled from a uniform random distribution between 50 and 100, but for the number of hidden layers $L$ it is more sensible to maybe explore all the values between 2 and 4. For the learning rate $\alpha$ a uniform random sampling would waste many computational resources since we want to sample a much bigger range [0.0001, 1], and a logarithmic random sampling would be more appropriate.


    

<figure id="fig:scales">
    <img src="{{site.baseurl}}/pages/ML-31-DeepLearningTuning_files/ML-31-DeepLearningTuning_9_0.png" alt="png">
    <figcaption>Figure 62. Random sampling in appropriate ranges and scales for different parameters. Uniform random sampling of no. of hidden units $^{[l]}$ in the range $[50, 100]$ (A); Complete exploration of integer no. of hidden layers $L$ in the range $[2, 4]$ (B); Logarithmic random sampling of learning rate $\alpha$ in the range $\left [10^{-4}, 10^0 \right ]$ (C)</figcaption>
</figure>.

A different strategy is usually employed when sampling for the hyperparameters for exponentially weighted averages. Common values for $\beta$ range from 0.9 to 0.999 but a linear sampling in this range would be extremely inefficient, since the effect of small changes in $\beta$ is almost null when $\beta \approx 0.9$ and very big when $\beta \approx 0.999$

$$
\begin{aligned}
& \beta = 0.9000 \to \frac{1}{1-\beta}\approx 10  \qquad & \to & \qquad \beta = 0.9005 \to \frac{1}{1-\beta}\approx 10 \\
\\
& \beta = 0.9990 \to \frac{1}{1-\beta}\approx 1000  \qquad & \to &\qquad  \beta = 0.9995 \to \frac{1}{1-\beta}\approx 2000 \\
\end{aligned}
$$

For this reason, usually instead of $\beta$, we sample $\beta - 1$ in a logarithmic random distribution in the range $\left[10^{-1}, 10^{-3} \right]$ (<a href="#fig:samplebeta">Figure 63</a>)


    

<figure id="fig:samplebeta">
    <img src="{{site.baseurl}}/pages/ML-31-DeepLearningTuning_files/ML-31-DeepLearningTuning_11_0.png" alt="png">
    <figcaption>Figure 63. Hyperparameter sampling of $1-\beta$ (exponentially weighted average parameter)</figcaption>
</figure>
