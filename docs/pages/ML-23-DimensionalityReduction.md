---
layout: default
title: "Unsupervised Learning - Dimensionality Reduction"
categories: Unsupervised
permalink: /ML23/
order: 23
comments: true
---

# Dimensionality Reduction
When training a learning algorithm on a training set with hundreds or thousands of features, it is very likely that some of them are redundant as in <a href="#redundantlen">Figure 22</a>. Two features do not need to be encoding for the same property to be redundant, it is sufficient that they are highly correlated.


    

<figure id="redundantlen">
    <img src="{{site.baseurl}}/pages/ML-23-DimensionalityReduction_files/ML-23-DimensionalityReduction_2_0.png" alt="png">
    <figcaption>Figure 22. Two redundant features (same property in different units of measure, the relationship not being perfectly linear is due to different approximation of the measurements) (A); Two redundant features measuring different but highly correlated properties (B); features space of panel A collapsed in (projected on) a single dimension (C) </figcaption>
</figure>

Dimensionality reduction is not limited to bi-dimensional data; a typical task of dimensionality reduction could be to reduce a $\mathbb{R}^{1000}$ to a $\mathbb{R}^{100}$ feature space

Dimensionality reduction is used for two main purposes: **data compression** and **data visualization**. Data compression is a label that covers a very wide range of uses, from simply occupying less virtual memory for files in hard disk to speed up learning algorithms; the use of dimensionality reduction for data visualization aims at reducing $n$ features to 2 or 3 features, which are the maximum number of dimensions in a plot.

##  Principal Component Analysis
Principal Component Analysis (PCA) is the most common algorithm for dimensionality reduction.

If we want to reduce data from 2 dimensions to 1 dimension (<a href="#pcaline">Figure 23</a>), the goal of PCA is to find a vector $u^{(1)} \in \mathbb{R^n}$ onto which to project the data so as to minimize the projection error.


    

<figure id="pcaline">
    <img src="{{site.baseurl}}/pages/ML-23-DimensionalityReduction_files/ML-23-DimensionalityReduction_5_0.png" alt="png">
    <figcaption>Figure 23. Distance calculation for PCA in the linear case (A) and linear regression (B).</figcaption>
</figure>

In $n$ dimensions PCA tries to find a surface with a smaller number of dimensions $k$ ($k$ vectors $u^{(1)}, u^{(2)}, \ldots, u^{(k)}$) on which to project the data so that the sum of squares of the projections distance (projection error) is minimized.

### PCA vs Linear Regression
While it may be cosmetically similar, there is substantial difference between PCA and linear regression. 

In linear regression (<a href="#pcaline">Figure 23</a>, panel B) we try to predict some value $y$ given some input feature $x_1$  and in training linear regression we try to minimize the **vertical distance** between $x_1, y$ points and a straight line.

In PCA (<a href="#pcaline">Figure 23</a>, panel A) there is no variable $y$: we try to reduce the dimensionality of a feature space $x_1, x_2$ by minimizing the **projection error** of feature points and a straight line.

## PCA algorithm
### Preprocessing
For PCA to work properly is essential to pre-process data. 

With **mean normalization** we replace each $x_j^{(i)}$ with $x_j - \mu_j$, where 

$$\mu_j = \frac{1}{m}\sum^m_{i=1}x_j^{(i)}$$

If different features are on different scales we need to **scale features** to a comparable range of values 

$$\frac{x_j - \mu_j}{s_j}$$

where $s_j$ is a measure of the range of values of $x_j$, commonly the standard deviation.

### Dimensionality reduction
The objective of PCA is to project a higher number of dimensions $n$ on a lower number of dimensions $k$ (defined by $k$ vectors $ \{u_1, u_2, \ldots, u_k \}$), as shown in <a href="#redundantlen">Figure 22</a>, panels A and C.

The mathematical proof of this process is rather complex but the procedure is instead quite simple.

The first step in the PCA algorithm is the calculation of the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) $\Sigma$, which will be an $n \times n$ matrix

$$
\Sigma = \frac{1}{m} \sum^n_{i=1} \left ( x^{(i)} \right )\left ( x^{(i)} \right )^T
$$

and then calculate the [eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of $\Sigma$. Since $\Sigma$ is a [symmetric positive definite matrix](https://en.wikipedia.org/wiki/Definite_symmetric_matrix) you usually calculate the eigenvectors with [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD). 

SVD outputs three matrices $U, S, V$, of which we are interested in the $U$ matrix. This will also be an $n \times n$ matrix, whose columns represents the vectors $u$.

$$
U=
\begin{bmatrix}
| & | &  & | \\
u^{(1)} & u^{(2)} & \cdots & u^{(n)} \\
| & | &  & | \\
\end{bmatrix} \in \mathbb{R}^{n \times n}
$$

To obtain the reduced space described by $k$ vectors $u$ we just take the first $k$ columns from the matrix $U$.

$$
U_\text{reduce}=
\begin{bmatrix}
| & | &  & | \\
u^{(1)} & u^{(2)} & \cdots & u^{(k)} \\
| & | &  & | \\
\end{bmatrix} \in \mathbb{R}^{n \times k}
$$

And the projection of features $x \in \mathbb{R}^n$ to components $z \in \mathbb{R}^k$ is calculated as 

$$
z = U_\text{reduce}^T x 
$$

and since $U_\text{reduce}^T \in \mathbb{R}^{k \times n}$ and $x \in \mathbb{R}^{n \times 1}$, then $z \in \mathbb{R}^{k \times 1}$ or $z \in \mathbb{R}^k$
