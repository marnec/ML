---
layout: default
title: "Deep Learning - Speed up learning - Normalization"
categories: deeplearning
permalink: /ML28/
order: 28
comments: true
---

# Training set normalization
Training set normalization is a technique that speeds up the learning process. Typically, input features come in range that differ of some order of magnitude. One feature might come in range $[0, 1000]$ and another in range $[0, 1]$. 

Difference in the scale of input features might make training very slow and for this reason input features are usually normalized. In <a href="#fig:meannorm">Figure 49</a> the effect of the two steps of **mean normalization** are shown.


    

<figure id="fig:meannorm">
    <img src="{{site.baseurl}}/pages/ML-28-DeepLearningNorm_files/ML-28-DeepLearningNorm_2_0.png" alt="png">
    <figcaption>Figure 49. mean normalization effect on feature space $X \in \mathbb{R}^{2}$. Raw feature space (A); $X - \mu$ (B); $\frac{X}{\sigma^2}$ (C)</figcaption>
</figure>.

To intuitively understand why mean normalization speeds up training the values, <a href="#costnorm">figure below</a> shown a simplified view of how the space of the values assumed by the cost function $J$ change with un-normalized (panels A,C) and normalized inputs (panels B,D)


    

<figure id="costnorm">
    <img src="{{site.baseurl}}/pages/ML-28-DeepLearningNorm_files/ML-28-DeepLearningNorm_4_0.png" alt="png">
    <figcaption>Figure 50. Representative shape of $J$ for un-normalized (A, C) and normalized (B, D) feature space</figcaption>
</figure>

## Vanishing or Exploding gradients
Under some circumstances it may happen, especially in very deep neural networks, that gradients (derivatives) assume very small (vanishing) or big (exploding) values.

This problem can be drastically reduced through a careful choice of the random weight initialization.

Suppose we have a very deep network as in the <a href="#fig:superdeep">Figure 51</a>. This network has many hidden layers and consequently a series of parameters matrices $(w^{[1]}, w^{[2]}, \dots w^{[L]})$. For the sake of simplicity let's say that this network has linear activation function ($g(z) = z$) and that $b^{[l]}=0$


    

<figure id="fig:superdeep">
    <img src="{{site.baseurl}}/pages/ML-28-DeepLearningNorm_files/ML-28-DeepLearningNorm_7_0.png" alt="png">
    <figcaption>Figure 51. A very deep neural network that can suffer from either exploding or vanishing gradients depending on the values of $w^{[l]}$</figcaption>
</figure>

For this network

$$
\begin{aligned}
\hat{y} = w^{[L]} \cdot w^{[L-1]} \cdot w^{[L-2]} \cdot \ldots \cdot w^{[2]} \cdot  w^{[1]} \cdot x \\
z^{[1]} = w^{[1]}x \\
a^{[1]} = g(z^{[1]}) = z^{[1]} \\
a^{[2]}  =  g(z^{[2]}) =  g(w^{[2]}a^{[1]}) \\
a^{[2]}  =  g(z^{[2]}) =  g(w^{[2]}w^{[1]}x) \\
\cdots
\end{aligned}
$$

Suppose that each of our weight matrices $w^{[l]}$ is

$$
w^{[l]} = 
\begin{bmatrix}
1.5 & 0 \\
0 & 1.5
\end{bmatrix}
\qquad \to \qquad 
\hat{y}=
\begin{bmatrix}
1.5 & 0 \\
0 & 1.5
\end{bmatrix}^{L-1}x
$$

$\hat{y}$ will increase exponentially with $1.5^L$. This implies that for large values of $L$ the value of $\hat{y}$ will explode. Conversely the value of $\hat{y}$ will vanish for weights $w^{[l]} < I$ (where $I$ is the identity matrix). 

In general 

$$
\hat{y} 
\begin{cases}
w^{[l]} > I \qquad \to \text{explode} \\ 
w^{[l]} < I \qquad \to \text{vanish}
\end{cases}
$$

And while we made this argument for the activation values we can make a similar argument for the gradients

## Correct weight initialization to attenuate gradient derive

Let's focus on a single neuron with 4 input features


```python
ax, *_ = ann([4, 1], node_labels=True, radius=2)
ax.set_aspect('equal')
```


    

<figure id="fig:onelayernn">
    <img src="{{site.baseurl}}/pages/ML-28-DeepLearningNorm_files/ML-28-DeepLearningNorm_10_0.png" alt="png">
    <figcaption>Figure 52. A single layer neural network with 4 input features</figcaption>
</figure>

For the network in <a href="#fig:onelayernn">Figure 52</a>

$$
z = w_1x_1 + w_2x_2 + \ldots + w_nx_n
$$

So the larger $n$ is, the smaller we want to set $w_i$ in order to prevent gradient explosion. It would be ideal to set the variance of $w_i$ to be inversely proportional to $n$

$$
\sigma^2(w_i) = \frac{1}{n}
$$

So, in order to set the variance for random variables **drawn from a Gaussian distribution** we would write our weights matrix as:


```python
nx = 4
np.random.randn(1, nx)*np.sqrt(1/nx)
```




    array([[-0.71975612,  0.02579541, -0.10470933, -0.35934415]])



where [`randn()`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html) draws from a Gaussian distribution and the term can vary depending on the activation function used and implementation details:

For a ReLU activation function:

$$
\sqrt{\frac{2}{n^{[l-1]}}}
$$

For $\tanh$

$$
\sqrt{\frac{1}{n^{[l-1]}}}
$$

Or sometimes this different implementation, also called Xavier initialization:

$$
\sqrt{\frac{2}{n^{[l-1]}+1}}
$$
