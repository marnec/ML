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
$$

Then we have

$$
\hat{y}=
\begin{bmatrix}
1.5 & 0 \\
0 & 1.5
\end{bmatrix}^{L-1}x
$$

And this means that for large values of $L$ the value of $\hat{y}$ will explode, it will increase exponentially with $1.5^L$.

Conversely the value of $\hat{y}$ will vanish for weights $w^{[l]} < I$ (where $I$ is the identity matrix). So we can say that the activation values (and consequently $\hat{y}$ will explode or vanish, respectively, for $w^{[l]} > I$ or $w^{[l]} < I$.

And while we made this argument for the activation values we can make a similar argument for the gradients
