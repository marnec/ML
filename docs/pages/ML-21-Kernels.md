---
layout: default
title: "SVM - Kernels"
categories: SVM
permalink: /ML21/
order: 21
comments: true
---

# Non-linear hypothesis with SVMs
In this section we will see how kernels enable SVMs to model complex non-linear data. Suppose we have a dataset like that in <a href="#ellipserandom">Figure 13</a>. 


    

<figure id="ellipserandom">
    <img src="{{site.baseurl}}/pages/ML-21-Kernels_files/ML-21-Kernels_3_0.png" alt="png">
    <figcaption>Figure 13. Non linear data scattered on its features space $(x_1, x_2)$. In this plots coordinates are randomly generated to fill two ellipses.</figcaption>
</figure>

We could manually design an hypothesis with high order polynomial features to model the data, but is there a better way to choose the features?

## Kernels
In this section we will start to explore the idea of **kernels** to define new features. 

Let's manually choose three points in the feature space $(x_1, x_2)$, that we will call the **landmarks** $l^{(1)}, l^{(2)}, l^{(3)}$ (<a href="#manuallandmarks">Figure 14</a>). We are going to define our as a measure of similarity between the training examples and the landmarks.


    

<figure id="manuallandmarks">
    <img src="{{site.baseurl}}/pages/ML-21-Kernels_files/ML-21-Kernels_6_0.png" alt="png">
    <figcaption>Figure 14. Three landmarks points chosen manually</figcaption>
</figure>

Given an example $x$, we will define three features $f_1, f_2, f_3$:

$$
\begin{align}
& f_1 = k(x, l^{(1)}) = \exp\left({-\frac{\| x - l^{(1)}\|^2}{2 \sigma^2}}\right) \\
& f_2 = k(x, l^{(2)}) = \exp\left({-\frac{\| x - l^{(2)}\|^2}{2 \sigma^2}}\right) \\
& f_3 = k(x, l^{(3)}) = \exp\left({-\frac{\| x - l^{(3)}\|^2}{2 \sigma^2}}\right)
\end{align}
$$

where $k$ stands for kernel and is a function that measure similarity and $\| x - l^{(i)}\|$ is the euclidean distance between the example $x$ and the landmark point $l^{(i)}$. In particular the specific kernel $k$ used here is called the **Gaussian kernel**.

Let's explore more in detail the first landmark, whose numerator can be also written as the square of the component-wise distance of the vector $x$ from the vector $l^{(1)}$:

$$f_1=k\left( x, l^{(i)}  \right) = \exp \left( -\frac{\left \| x - l^{(1)} \right \|^2}{2\sigma^2}\right) = \exp \left( -\frac{\sum_{j=1}^n \left(x_j-l_j^{(1)} \right)^2}{2\sigma^2} \right)$$

When $x \approx l^{(1)}$, the euclidean distance between $x$ and $l^{(1)}$ will be $\approx 0$ and $f_1 \approx 1$ 

$$
f_1 \approx \exp \left ( - \frac{0^2}{2 \sigma^2} \right ) \approx 1 
$$

Conversely, when $x$ is far from $l^{(1)}$ the euclidean distance between $x$ and $l^{(1)}$ will be a large number and $f_1 \approx 0$

$$
f_1 = \exp \left( - \frac{\text{large number}^2}{2 \sigma^2} \right) \approx 0
$$

To summarize:

$$
f_1 \approx
\begin{cases}
1 & \text{if } x \approx l^{(1)} \\
0 & \text{if } x \neq l^{(1)}
\end{cases}
$$

Since we have three landmark points $l^{(1)}, l^{(2)}, l^{(3)}$, given an example $x$ we can define three new features $f_1, f_2, f_3$.

### Similarity function
Let's take a look at the similarity function $k$. Suppose 

$$
l^{(1)} = \begin{bmatrix}3\\5\end{bmatrix}, \quad f_1 = \exp \left( -\frac{\left \| x - l^{(1)} \right \|^2}{2\sigma^2}\right)
$$

By looking at <a href="#gaussk">Figure 15</a>, we can see how $f_1 \to 1$ when $x_1 \to 3$ and $x_2 \to 5$, where $(3, 5)$ are the coordinates of $l^{(1)}$. 


    

<figure id="gaussk">
    <img src="{{site.baseurl}}/pages/ML-21-Kernels_files/ML-21-Kernels_9_0.png" alt="png">
    <figcaption>Figure 15. Guassian kernel</figcaption>
</figure>

It is also interesting to notice the effect of different values of $\sigma^2$ on $f_1$ output. When $\sigma^2$ decreases, $f_1$ falls to $0$ much more rapidly when $x$ stray from $l^{(1)}$. Conversely if the value of $\sigma^2$ is large the decay of $f_1$ is much slower.