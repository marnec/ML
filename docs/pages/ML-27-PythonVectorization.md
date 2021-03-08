---
layout: default
title: "Python vectorization"
categories: deeplearning
permalink: /ML27/
order: 27
comments: true
---

# Python vectorization
In the pre-deep-learning era vectorization was optional, in the deep-learning era vectorization absolutely necessary since both the size of networks and of data is vastly increased.

## Vector-vector product
In particular, in deep learning (and in machine learning in general) we need to calculate 

$$
z = w^Tx+b
$$

for 

$$
w =
\begin{bmatrix}
\vdots \\ \vdots
\end{bmatrix} \in \mathbb{R}^{n_x}
\qquad 
x = \begin{bmatrix}
\vdots \\ \vdots
\end{bmatrix} \in \mathbb{R}^{n_x}
$$

The vectorized form of this operation in python is 


```python
np.dot(w, x) + b
```

where `np.dot(w, x)` $\equiv w^Tx$

## Matrix-vector product
Incidentally, the matrix-vector product $Av$, where 

$$
A = \begin{bmatrix}
\ddots &  \\
&   \\
&  \ddots \\
\end{bmatrix} \in \mathbb{R}^{m \times n} \qquad 
v=\begin{bmatrix}
\vdots \\ \vdots
\end{bmatrix} \in \mathbb{R}^n
$$


```python
np.dot(A, v)
```

Notice that the exact same syntax performs both vecto-vector and matrix-vector multiplication, this is due to the overload implemented in the `np.dot` function. To know more about it, check out [its documentation](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)

## Vectorized element-wise operations
To apply a function element by element to whole arrays you can simply use`np.ufuncs` ([numpy universal functions](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html#numpy.ufunc))


```python
v
```




    array([0.34, 0.95, 0.27, 0.29, 0.14, 0.03, 0.14, 0.97, 0.73, 0.98])




```python
np.exp(v).round(2)
```




    array([1.4 , 2.59, 1.31, 1.34, 1.15, 1.03, 1.15, 2.64, 2.08, 2.66])




```python
np.log(v).round(2)
```




    array([-1.08, -0.05, -1.31, -1.24, -1.97, -3.51, -1.97, -0.03, -0.31,
           -0.02])




```python
v + 1
```




    array([1.34, 1.95, 1.27, 1.29, 1.14, 1.03, 1.14, 1.97, 1.73, 1.98])




```python
v * 2
```




    array([0.68, 1.9 , 0.54, 0.58, 0.28, 0.06, 0.28, 1.94, 1.46, 1.96])


