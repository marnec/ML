---
layout: default
title: "Neural Networks - Motivation"
categories: neuralNetwork
permalink: /ML12/
order: 12
---

# Non-linear hypothesis
Imagine that you have a classification problem and your training set has 100 features

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
\dots\\
x_{100}
\end{bmatrix}
$$

Let's say that the data happens to be complex; we approach this problem with logistic regression and we are aware that our hypothesis is non-linear and should includes many polynomial features

$$
\begin{align}
&g(\theta_0 + \theta_1x_1 + \theta_2x_2 \\
&\theta_3x_1x_2 + \theta_4x_1^2x_2 \\
&\theta_5x_1^3x_2 + \theta_6x_1x_2^2 + \dots )
\end{align}
$$

Since $n=100$ we have an enormous amount of possible polynomial combinations. The number of polynomial combination will, in fact, be $\approx \frac{n^2}{2}$. We will have 5000 possible second order polynomials $(x_1x_2, x_1x_3, x_1x_4, \dots)$ and 170,000 possbile third order polynomials. 

If we include second order polynomials were we take only the squares of features $(x_1^2, x_2^2, \dots)$ we would have only 100 second order features but we could only model simple decision boundaries (circles, ellipsis) and we would exclude a-priori more complex models.

The number of possible features blows up very quickly by increasing the order of the polynomials so this doesn't look like a good way to increase the number of features of our model.

So we can't use logistic regression for highly dimensional non-linear problems. Neural networks are instead quite useful in these cases.

# Neural Networks
## Origin of neural networks
Neural networks are a pretty old algorith the tries to rundimentally mimic the brain functioning. They were widely used in the '80s and early '90s but their populariy diminishider in the late '90s. They had a recent resurgence and they are now a state-of-the-art technique for many applications.

One of the reason for their resurgence was that, being computationally expensive, only recently computers became fast enough to really use large-scale neural networks.

The idea that inspired neural networks is that our brain can perform many different operations (process images, sound and sense of touch, coordinate movement, perform numerical calculations) with *one learning algorithm* instead that with many dedicated programs.
