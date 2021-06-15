---
layout: default
title: "Preparatory concepts"
categories: introduction
permalink: /ML1/
order: 1
comments: true
---

# Derivatives and computation graph
Derivatives are a fundamental concept in machine learning, they are the building block of optimization and having an understanding of what a derivative is vastly helps in understanding how optimization and gradient descent work.

## Derivative
Suppose we have a function $f(a) = 3a$, then $f(2) = 6$. If we take a small increment of $a$ ($a'$) we will have $f(2.001) = 6.003$. Connecting $a$ and $a'$ forms a triangle, with an height ($a'-a$) and a width ($f(a') - f(a)$) (<a href="#fig:derivative">figure below</a>).

The slope $\frac{\text{height} }{\text{width}}=3$ so we say that the derivative of $f(a)$ at the point $a=2$ is $3$. Height and width are the the vertical and horizontal distances and the slope is also expressed as $\frac{df(a)}{da}$ or as $\frac{d}{da}f(a)$. The reason why $a'$ doesn't appear in this representation is because, formally, the derivative is calculated at a very small increment of $a$ such as $a' \approx a$.

For a straight line (<a href="#fig:derivative">figure below</a>, panel A) the derivative is constant along the whole line.


    
![png](ML-1-Preparatory_files/ML-1-Preparatory_2_0.png)
    


<i id="fig:derivative">The concept of derivative applied to a straight line (A), where the derivative is constant along the whole length of the function; and to a non-linear function (B), where the derivative changes based on the value of $a$.</i>

## Computational graph
The computational graph explains the forward- and backward- propagation (as to say the flow of the computation) that takes place in the training of a neural network. 

To illustrate the computation graph let's use a simpler example than a full blown neural network, let's say that we are writing a function $J(a, b, c) = 3(a+bc)$. In order to compute this function there are three steps: 

1. $u = bc$
2. $v = a + u$
3. $J=3v$

We can draw these steps in a computational graph (<a href="#compgraph">figure below</a>)


    
![png](ML-1-Preparatory_files/ML-1-Preparatory_4_0.png)
    


<i id="compgraph">Computational graph showing the flow of a very simple process</i>

Suppose we want to calculate $\frac{dJ}{dv}$ ( in other words if we change the value $v$ of a little amount how would the value of $J$ change?). 

* $J = 3v$
* $v = 11 \to 3.001$
* $J = 33 \to 33.003$

So 

$$\frac{dJ}{dv}=\frac{0.003}{0.001}=3$$

In the terminology of backpropagation if we want to compute $\frac{dJ}{dv}$ we take one step back from $J$ to $v$


    
![png](ML-1-Preparatory_files/ML-1-Preparatory_6_0.png)
    


We now want to calculate $\frac{dJ}{da}$, in other words the change of value $J$ when $a$ changes

* $a = 5 \to 5.001$
* $v = 11 \to 11.001$
* $J = 33 \to 33.003$

So, once again

$$\frac{dJ}{da}=\frac{0.003}{0.001}=3$$

Where the net change is given by 

$$
\frac{dJ}{da}=\frac{dJ}{dv}\frac{dv}{da}
$$


In calculus this is called the **chain rule** where $a$ affects $v$ that affects $J$ ($a\to v \to J$). So that the change of $J$ when $a$ is given by the product $\frac{dJ}{dv}\frac{dv}{da}$. This illustrates how having computed $\frac{dJ}{dv}$ helps in calculating $\frac{dJ}{da}$


    
![png](ML-1-Preparatory_files/ML-1-Preparatory_8_0.png)
    


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




    array([0.13, 0.98, 0.16, 0.9 , 0.73, 0.39, 0.46, 0.01, 0.25, 0.69])




```python
np.exp(v).round(2)
```




    array([1.14, 2.66, 1.17, 2.46, 2.08, 1.48, 1.58, 1.01, 1.28, 1.99])




```python
np.log(v).round(2)
```




    array([-2.04, -0.02, -1.83, -0.11, -0.31, -0.94, -0.78, -4.61, -1.39,
           -0.37])




```python
v + 1
```




    array([1.13, 1.98, 1.16, 1.9 , 1.73, 1.39, 1.46, 1.01, 1.25, 1.69])




```python
v * 2
```




    array([0.26, 1.96, 0.32, 1.8 , 1.46, 0.78, 0.92, 0.02, 0.5 , 1.38])



## Broadcasting
To a complete guide to broadcasting check out [numpy great documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html#:~:text=The%20term%20broadcasting%20describes%20how,that%20they%20have%20compatible%20shapes.&text=NumPy%20operations%20are%20usually%20done,element%2Dby%2Delement%20basis.)




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




    array([0.70273802, 0.59057133, 0.92142553, 0.38603558, 0.37320026])



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




    array([0.70273802, 0.59057133, 0.92142553, 0.38603558, 0.37320026])



and the inner product of `a` and `a.T` is not a matrix instead is a scalar


```python
np.dot(a, a.T)
```




    1.9799421335211578



So, instead of using rank 1 vectors you may want to use rank 2 vectors, which have a much more predictable behavior.


```python
a = np.random.rand(5, 1)
a
```




    array([[0.5600209 ],
           [0.05814544],
           [0.98815322],
           [0.24366541],
           [0.16702286]])




```python
a.T
```




    array([[0.5600209 , 0.05814544, 0.98815322, 0.24366541, 0.16702286]])




```python
np.dot(a, a.T)
```




    array([[0.31362341, 0.03256266, 0.55338646, 0.13645772, 0.09353629],
           [0.03256266, 0.00338089, 0.0574566 , 0.01416803, 0.00971162],
           [0.55338646, 0.0574566 , 0.97644678, 0.24077876, 0.16504418],
           [0.13645772, 0.01416803, 0.24077876, 0.05937283, 0.04069769],
           [0.09353629, 0.00971162, 0.16504418, 0.04069769, 0.02789664]])



rank 1 arrays can always be reshaped in row or columns vectors (or higher dimensional matrices)


```python
a = np.random.rand(5)
a
```




    array([0.63684256, 0.57294573, 0.41888045, 0.70994363, 0.3930198 ])




```python
a.reshape(5, 1)
```




    array([[0.63684256],
           [0.57294573],
           [0.41888045],
           [0.70994363],
           [0.3930198 ]])




```python
a.reshape(1, 5)
```




    array([[0.63684256, 0.57294573, 0.41888045, 0.70994363, 0.3930198 ]])


