---
layout: default
title: "Logistic Regression - Cost function"
categories: logisticRegression
permalink: /ML6/
order: 6
comments: true
---


    
![svg](ML-6-LogisticRegressionCostFunction_files/ML-6-LogisticRegressionCostFunction_1_0.svg)
    


The Cost function for a single training example is called the **Loss function** $\mathcal{L}$

$$\frac{1}{2}\left(h_\theta\left(x^{(i)}\right)-y^{(i)}\right)^2 \equiv \mathcal{L} \implies J(\theta)=\frac{1}{m}\sum_{i=1}^m\mathcal{L}$$

## Loss Function
For a single training example, the Cost function is called the **Loss function** $\mathcal{L}$

$$
\begin{equation*}
\mathcal{L}=
\begin{cases}
-log(h_\theta(x)), & \text{if } y=1\\
-log(1-h_\theta(x)), & \text{if } y=0\\
\end{cases}
\end{equation*}
$$



    
![svg](ML-6-LogisticRegressionCostFunction_files/ML-6-LogisticRegressionCostFunction_4_0.svg)
    


This cost function has some desirable properties:

* For $y=1$
    * $h_\theta(x)\to1 \implies \mathcal{L}\to0$. This is what we want because there should be an increasingly smaller cost when $h_\theta(x)\to y$ and no cost when $h_\theta(x) = y = 1$

    * $h_\theta(x) \to 0 \implies \mathcal{L}\to\infty$. This captures the intuition that if $y=1$ and $P(y=1\mid x;\theta)$ the algorithm is penalized by a large cost
    
* For $y=0$
    * $h_\theta(x)\to1 \implies \mathcal{L}\to\infty$. There is a big cost associated to predicting 1 when $y=0$
    * $h_\theta(x)\to0 \implies \mathcal{L}\to0$. There is no cost associated to predicting 0 when $y=0$

## Simplified cost function
Since $y\in{0,1}$ we can write the $\mathcal{L}$ function in a simpler way and compress the two cases in one equation.

$$
\mathcal{L}(h_\theta(x),y)=-y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))
$$

When $y=1$:

$$
\begin{align}
\mathcal{L}(h_\theta(x),y)&=-1\cdot\log(h_\theta(x))-0\cdot\log(1-h_\theta(x))\\
&=-\log(h_\theta(x))
\end{align}
$$

When $y=0$:

$$
\begin{align}
\mathcal{L}(h_\theta(x),y)&=-0\cdot\log(h_\theta(x))-1\cdot\log(1-h_\theta(x))\\
&=-\log(1-h_\theta(x))
\end{align}
$$

## Cost function
Now that we have a more compact way of writing the cost function we can write it for the whole dataset

$$
\begin{align}
J(\theta)&=\frac{1}{m}\sum_{i=1}^m\mathcal{L}\left(h_\theta \left(x^{(i)} \right),y^{(i)}\right)\\
&=-\frac{1}{m}\left[\sum_{i=1}^my^{(i)}\log\left(h_\theta(x^{(i)})\right)+\left(1-y^{(i)}\right)\log\left(1-h_\theta(x^{(i)})\right)\right]
\end{align}
$$

Although there are other cost functions that can be used this cost function can be derived from statistics using the principle of [maximum likelihood estimation.](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)

In order to estimate the parameters with this cost function we have to find the parameters $\theta$ that minimize $J(\theta)$

# Gradient descent
To minimize $J(\theta)$ we are going to use gradient descent

$$
\begin{equation*}
\theta_j := \theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)
\end{equation*}
\label{eq:gdescent} \tag{1}
$$

Where we repeat $\eqref{eq:gdescent}$ for all element $\theta_j$ of the parameters vector $\theta$ updating the parameters simoultanueously (after calculating them).

Deriving the term $\theta_j$ we have:

$$
\frac{\partial}{\partial\theta_j}=\frac{1}{m}\sum_{i=1}^m\left(h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}
$$

Plugging the derived term into the gradient descent we obtain

$$\theta_j := \theta_j-\frac{\alpha}{m}\sum_{i=1}^m\left(h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}$$

This looks identical to gradient descent for linear regression, however the definition of $h_\theta(x)$ is changed and is now $\frac{1}{1+e^{\theta^Tx}}$

# Vectorization
Vectorized implementations of the cost function and the gradient descent are

## Cost function

$$
\begin{align}
&h = g(\theta x)\\
&J(\theta)=\frac{1}{m}\left(-y^T\log(h)-(1-y)^T\log(1-h)\right)
\end{align}
$$

## Gradient descent

$$
\theta:=\theta-\frac{\alpha}{m}X^T\left(g(X\theta)-\vec{y}\right)
$$

# Advanced Optimization
By applying some concepts of optimization, we can fit logistic regression parameters much more efficiently than gradient descent and make the logistic regression algorithm scale better for large datasets.

Until now we have chosen to use the gradient descent optimization algorithm. However, there are other, more sophisticated optimization algorithms:

* [Conjugate descent](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
* [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
* [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)

These algorithms, at the cost of being more complex, share a series of advantages:

* They remove need of manually picking an $\alpha$
* They are often faster than gradient descent
