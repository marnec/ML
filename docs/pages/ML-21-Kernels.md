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

### Multiple landmarks
Let's get back to <a href="#manuallandmarks">Figure 14</a> and let's try to put together what we have learned in the case of multiple landmarks.

We have three features $f_1, f_2, f_3$ and our hypothesis:

$$
y = 1 \quad \text{if} \quad \theta_0 + \theta_1f_1 + \theta_2f_2 + \theta_3f_3 \geq 0
$$

Suppose that we have run our learning algorithm and came up with the following parameters

$$
\theta=
\begin{bmatrix}
0.5 \\ 1 \\ 1 \\ 0
\end{bmatrix}
$$

Now let's consider various training examples placed on the features space $(x_1, x_2)$ represented on <a href="#manuallandmarks2">Figure 16</a>.

Training example $x^{(1)}$ is close to $l^{(1)}$, hence $f_1 \approx 1$ while $f_2, f_3 \approx 0$. Its prediction will be:

$$
\begin{align}
h_\theta(x^{(1)}) & =  \theta_0 + \theta_1 \cdot 1 + \theta_2 \cdot 0 + \theta_3 \cdot 0 \\
&= -0.5 + 1 \\
&= 0.5 \geq 1 \\
& \to y^{(1)} = 1  
\end{align}
$$


Training example $x^{(2)}$ is far from any landmark, hence $f_1, f_2, f_3 \approx 0$ and $y^{(2)} \approx -0.5$




```python
fig, ax = plt.subplots()
ax.scatter(*a.T)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
for i, p in enumerate(a, 1):
    ax.text(*p + .01, '$l^{{({})}}$'.format(i), fontsize=13)

ax.plot(*a[0] + [.05, -.05], ls='none', marker='o', c='C1', label='$y^{(1)}=1$')
ax.text(*a[0] + [.07, -.04], '$x^{(1)}$', fontsize=13)
ax.plot(*a[1] + [.01, -.3], ls='none', marker='o', c='C9', label='$y^{(2)}=0$')
ax.text(*a[1] + [.03, -.29], '$x^{(2)}$', fontsize=13)
ax.legend(fontsize=13);
```


    

<figure id="manuallandmarks2">
    <img src="{{site.baseurl}}/pages/ML-21-Kernels_files/ML-21-Kernels_12_0.png" alt="png">
    <figcaption>Figure 16. Manually placed landmarks as in <a href="#manuallandmarks">Figure 14</a> and training example falling close or far from landmarks.</figcaption>
</figure>
