---
layout: default
title: "Python vectorization"
categories: deeplearning
permalink: /ML25/
order: 25
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




    array([0.28, 0.72, 0.5 , 0.39, 0.84, 0.17, 0.14, 0.96, 0.02, 0.75])




```python
np.exp(v).round(2)
```




    array([1.32, 2.05, 1.65, 1.48, 2.32, 1.19, 1.15, 2.61, 1.02, 2.12])




```python
np.log(v).round(2)
```




    array([-1.27, -0.33, -0.69, -0.94, -0.17, -1.77, -1.97, -0.04, -3.91,
           -0.29])




```python
v + 1
```




    array([1.28, 1.72, 1.5 , 1.39, 1.84, 1.17, 1.14, 1.96, 1.02, 1.75])




```python
v * 2
```




    array([0.56, 1.44, 1.  , 0.78, 1.68, 0.34, 0.28, 1.92, 0.04, 1.5 ])



## Broadcasting
To a complete guide to broadcasting check out [numpy great documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html#:~:text=The%20term%20broadcasting%20describes%20how,that%20they%20have%20compatible%20shapes.&text=NumPy%20operations%20are%20usually%20done,element%2Dby%2Delement%20basis.)


```python
A = pd.DataFrame([[56, 0, 4.4, 6.8], [1.2, 104, 52, 8], [1.8, 135, 99, 0.9]], 
                        columns=['Apples', 'Beef', 'Eggs', 'Potatoes'], index=['Carb', 'Protein', 'Fat'])
A
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apples</th>
      <th>Beef</th>
      <th>Eggs</th>
      <th>Potatoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Carb</th>
      <td>56.0</td>
      <td>0</td>
      <td>4.4</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>Protein</th>
      <td>1.2</td>
      <td>104</td>
      <td>52.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Fat</th>
      <td>1.8</td>
      <td>135</td>
      <td>99.0</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
A = A.values
A
```




    array([[ 56. ,   0. ,   4.4,   6.8],
           [  1.2, 104. ,  52. ,   8. ],
           [  1.8, 135. ,  99. ,   0.9]])




```python
cal = A.sum(axis=0)
cal
```




    array([ 59. , 239. , 155.4,  15.7])




```python
(A / cal.reshape(1, 4) * 100)
```




    array([[94.91525424,  0.        ,  2.83140283, 43.31210191],
           [ 2.03389831, 43.51464435, 33.46203346, 50.95541401],
           [ 3.05084746, 56.48535565, 63.70656371,  5.73248408]])




```python
A / cal * 100
```




    array([[94.91525424,  0.        ,  2.83140283, 43.31210191],
           [ 2.03389831, 43.51464435, 33.46203346, 50.95541401],
           [ 3.05084746, 56.48535565, 63.70656371,  5.73248408]])



In general if you have a $m, n$ matrix (A) 

* if you apply an operation with an $1, n$ matrix (B), then B will be copied $m$ times and the operations applied element-wise
* if you apply an operation with an $m, 1$ matrix (C), then C will be copied $n$ times and the operations applied element-wise

## numpy Vectors
`numpy` offers great flexibility at the cost of rigorousness, sometimes wrong-looking expression give unexpectedly correct results and vice versa.
Heres a series of considerations and suggestions for dealing with `numpy`.

For example let's take a random vector of 5 elements


```python
a = np.random.rand(5)
a
```




    array([0.58793071, 0.57107289, 0.0889954 , 0.04943523, 0.2427539 ])



Whose shape is


```python
a.shape
```




    (5,)



This is called a rank 1 vector in python and it's neither a row vector nor a column vector and its behavior is sometimes unexpected. 

For example, its transpose is equal to itself 


```python
a.T
```




    array([0.58793071, 0.57107289, 0.0889954 , 0.04943523, 0.2427539 ])



and the inner product of `a` and `a.T` is not a matrix instead is a scalar


```python
np.dot(a, a.T)
```




    0.7410802482964651



So, instead of using rank 1 vectors you may want to use rank 2 vectors, which have a much more predictable behavior.


```python
a = np.random.rand(5, 1)
a
```




    array([[0.09125468],
           [0.93066971],
           [0.74560987],
           [0.51267681],
           [0.45281322]])




```python
a.T
```




    array([[0.09125468, 0.93066971, 0.74560987, 0.51267681, 0.45281322]])




```python
np.dot(a, a.T)
```




    array([[0.00832742, 0.08492797, 0.06804039, 0.04678416, 0.04132133],
           [0.08492797, 0.8661461 , 0.69391652, 0.47713278, 0.42141955],
           [0.06804039, 0.69391652, 0.55593408, 0.38225689, 0.33762201],
           [0.04678416, 0.47713278, 0.38225689, 0.26283752, 0.23214684],
           [0.04132133, 0.42141955, 0.33762201, 0.23214684, 0.20503981]])



rank 1 arrays can always be reshaped in row or columns vectors (or higher dimensional matrices)


```python
a = np.random.rand(5)
a
```




    array([0.61699538, 0.19953253, 0.12773572, 0.98264147, 0.05461118])




```python
a.reshape(5, 1)
```




    array([[0.61699538],
           [0.19953253],
           [0.12773572],
           [0.98264147],
           [0.05461118]])




```python
a.reshape(1, 5)
```




    array([[0.61699538, 0.19953253, 0.12773572, 0.98264147, 0.05461118]])


