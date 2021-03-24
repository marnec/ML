---
layout: default
title: "Neural Networks - Model Representation"
categories: neuralNetwork
permalink: /ML10/
order: 10
comments: true
---

# Model Representation of Neural Network
Anatomical neurons are cells that are present in the brain in millions. A neuron has a cell body, a number of input wires, called *dendrites* and an output wire called *axon*.


    
![png](ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_2_0.png)
    


In a simplistic way a neuron is a computational unit that receive some input via dendrites, does some computation and then outputs something via the axon to other neurons in the brain.

A neuron implemented on the computer has a very simple model that mimics the architecture of an anatomical neuron. We're a going to model a neuron as just a logistic unit. The yellow node represents the *body* of the neuron, which is fed input through its *dendrites*, and produces an output $h_\theta(x)$ that is produced by the neuron body, though its **activation function** and transported forward by the neuron *axon*. Where $h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}$


    
![png](ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_4_0.png)
    


A simpler representation is sometimes used to depict a neural network

$$
[x_0x_1x_2x_3]\to[]\to h_\theta(x)
$$

where $x_0$ (sometimes called the bias unit) is usually omitted in favor of another representation:

$$
[x_1x_2x_3]\to[]\to h_\theta(x)
$$

and the parameters, or **weights** ($w$) are accompanied by the **bias** $b$.

Until now we represented single neurons; a neural network is a group of different neurons connected together. The input nodes are grouped in what is called the **input layer** ($x$), which is always the first layer of the neural network. The final layer is called the **output layer**, since it computes the final value of our hypothesis. And all layers in between the input and the output layers are called **hidden layers**. They are called hidden layers because we can't observes the values computed by these nodes.


    

<figure id="fig:simpleann">
    <img src="{{site.baseurl}}/pages/ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_7_0.png" alt="png">
    <figcaption>Figure 8. A simple neural network with one hidden layer</figcaption>
</figure>

The computational entities in a neural networks are:

* $a_i^{[l]}$  activation neuron/unit $i$ in layer $l$
* $W^{[l]}$ matrix of weights controlling the function mapping from layer $l$ to layer $l+1$ 
* $b^{[l]}$ the bias vectors

And the computation in the network

$$
\left[x_1 x_2 x_3 \right]\to \left[a_1^{[1]}a_2^{[1]}a_3^{[1]} \right]\to a_1^{[2]} \equiv \hat{y}
$$

# Forward propagation

The flow of the computation in the network in <a href="#fig:simpleann">Figure 8</a> from input (left) to prediction (right), called forward propagation, is just like that in logistic regression but a lot more times. In fact, each unit in layer $l$ is **densely connected** (namely is connected to all units in layer $l+1$) and we will have to compute a logistic regression for each connection.

So, for example, the computations that we will have to execute from the input layer to the first layer will be:

$$
\begin{align}
& a_1^{[1]} =
g \left(
\left( W_{11}^{[1]}x_1 + b^{[1]}_{11} \right) + 
\left( W_{12}^{[1]}x_2 + b^{[1]}_{12} \right) +
\left( W_{13}^{[1]}x_3 + b^{[1]}_{13} \right)
\right) \\
& a_2^{[1]} = 
g \left(
\left( W_{21}^{[1]}x_1 + b^{[1]}_{21} \right) + 
\left( W_{22}^{[1]}x_2 + b^{[1]}_{22} \right) +
\left( W_{23}^{[1]}x_3 + b^{[1]}_{23} \right)
\right) \\
& a_3^{[1]} = 
g \left(
\left( W_{31}^{[1]}x_1 + b^{[1]}_{31} \right) + 
\left( W_{32}^{[1]}x_2 + b^{[1]}_{32} \right) +
\left( W_{33}^{[1]}x_3 + b^{[1]}_{33} \right)
\right) \\
\end{align}
\label{eq:neuralnet} \tag{1}
$$

That is to say that we compute our hidden units in the first layer as a $3\times 3$ matrix of parameters $W^{[l]}_{ij}$, weighting the connection from unit $j$ in layer $l-1$ to unit $i$ in layer $l$.






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
      <th>$x_1$</th>
      <th>$x_2$</th>
      <th>$x_3$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>$a^{[1]}_1$</th>
      <td>$W^{[1]}_{11}$</td>
      <td>$W^{[1]}_{12}$</td>
      <td>$W^{[1]}_{13}$</td>
    </tr>
    <tr>
      <th>$a^{[1]}_2$</th>
      <td>$W^{[1]}_{21}$</td>
      <td>$W^{[1]}_{22}$</td>
      <td>$W^{[1]}_{23}$</td>
    </tr>
    <tr>
      <th>$a^{[1]}_3$</th>
      <td>$W^{[1]}_{31}$</td>
      <td>$W^{[1]}_{32}$</td>
      <td>$W^{[1]}_{33}$</td>
    </tr>
  </tbody>
</table>
</div>



## Vectorization

### First step of vectorization
Let's see vectorization for $\eqref{eq:neuralnet}$: this process can be then applied to any other layer. In  $\eqref{eq:neuralnet}$ we have the equations $\eqref{eq:l1unit1vect}$ are the operations required to calculate $a^{[1]}_1$ from input $x_1, x_2, x_3$ (<a href="#fig:annfirststeps">Figure 9</a>, panel A).

$$
\begin{align}
&z^{[1]}_1=w^{[1]T}_1x+b_1^{[1]} \\
&a^{[1]}_1=\sigma(z^{[1]}_1)
\end{align}
\label{eq:l1unit1vect} \tag{3}
$$



Similarly, $\eqref{eq:l1unit2vect}$ are the operations required to calculate $a^{[1]}_2$ from input $x_1, x_2, x_3$ (<a href="#fig:annfirststeps">Figure 9</a>, panel B)

$$
\begin{align}
&z^{[1]}_2=w^{[1]T}_2x+b_2^{[1]} \\
&a^{[1]}_2=\sigma(z^{[1]}_2)
\end{align}
\label{eq:l1unit2vect} \tag{5}
$$

Finally, $\eqref{eq:l1unit3vect}$  are the operations required to calculate $a^{[1]}_3$ from input $x_1, x_2, x_3$ (<a href="#fig:annfirststeps">Figure 9</a>, panel C)

$$
\begin{align}
&z^{[1]}_3=w^{[1]T}_3x+b_2^{[1]} \\
&a^{[1]}_3=\sigma(z_3^{[1]})
\end{align}
\label{eq:l1unit3vect} \tag{6}
$$


    

<figure id="fig:annfirststeps">
    <img src="{{site.baseurl}}/pages/ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_12_0.png" alt="png">
    <figcaption>Figure 9. First three steps of forward propagation to calculate the hidden units of the first layer from the input layer.</figcaption>
</figure>

### Second step of vectorization
Given the set of equation that describe the activtion of the first layer from the input layer, let's see how to calculate $z^{[1]}$ as a vector:

$$
\begin{align}
&z_1^{[1]} = w^{[1]T}_1 x + b_1^{[1]} & a_1^{[1]} = \sigma \left(z_1^{[1]}\right) \\
&z_2^{[1]} = w^{[1]T}_2 x + b_2^{[1]} & a_2^{[1]} = \sigma \left(z_2^{[1]}\right) \\
&z_3^{[1]} = w^{[1]T}_3 x + b_3^{[1]} & a_3^{[1]} = \sigma \left(z_3^{[1]}\right)
\end{align}
$$

Let's first take the vectors $w^{[1]T}_i$ and stack them into a matrix $W^{[1]}$. 

$w^{[1]}_i$ are column vectors, so their transpose are row vectors that are stacked vertically. 

$$
z^{[1]}=
\underbrace{
\begin{bmatrix}
\rule[.5ex]{2.5ex}{0.5pt} & w^{[1]T}_1 &\rule[.5ex]{2.5ex}{0.5pt}\\
\rule[.5ex]{2.5ex}{0.5pt} & w^{[1]T}_2 & \rule[.5ex]{2.5ex}{0.5pt}\\
\rule[.5ex]{2.5ex}{0.5pt} & w^{[1]T}_3 & \rule[.5ex]{2.5ex}{0.5pt}\\
\end{bmatrix}}_{s_j \times n}
\underbrace{
\begin{bmatrix}
x_1 \\ x_2 \\ x_3
\end{bmatrix}}_{n \times 1}+
\underbrace{
\begin{bmatrix}
b_1^{[1]} \\ b_2^{[1]} \\ b_3^{[1]}
\end{bmatrix}}_{s_j \times 1}
=\underbrace{\begin{bmatrix}
z_1^{[1]} \\ z_2^{[1]} \\ z_3^{[1]}
\end{bmatrix}}_{s_j \times 1}
$$

And now we can calculate $a^{[1]}$

$$
a^{[1]}=\sigma \underbrace{ \left( \begin{bmatrix}
z_1^{[1]} \\ z_2^{[1]} \\ z_3^{[1]}
\end{bmatrix} \right) }_{s_j \times 1}
$$

So, summarizing what we have written above and extending it to the second layer we have

$$
\begin{aligned}
&z^{[1]}= W^{[1]}x + b^{[1]} \\
&a^{[1]} = \sigma(z^{[1]})\\
&z^{[2]}= W^{[1]}a^{[1]} + b^{[2]}\\
&a^{[2]} = \sigma(z^{[2]})\\
\end{aligned}
\label{eq:vectanneqs} \tag{7}
$$

or more generally

$$
\begin{align}
&z^{[j]}=W^{[j]}a^{[j-1]} + b^{[j]} \\
&a^{[j]} = \sigma \left(z^{[j]}\right)
\end{align}
$$

### Third step of vetorization across multiple examples
The process in $\eqref{eq:vectanneqs}$ must be repeated for each training example $x^{(m)}$ and will produce $m$ outputs $a^{(m)[2]} = \hat{y}^{(m)}$

In a non-vectorized implementation you would have something along the lines of:

```python
for i in len(examples):
    z1[i] = w1 @ x[i] + b[i]
    a1[i] = sigmoid(z1[i])
    z2[i] = w2 @ x([i] + b[i]
    a2[i] = sigmoid(z1[i])
```

Given our vector of training examples:

$$
X=
\begin{bmatrix}
& \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & & \rule[-1ex]{0.5pt}{2.5ex}\\
&x^{(1)}&x^{(2)}&\dots&x^{(m)}\\
& \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & & \rule[-1ex]{0.5pt}{2.5ex}\\
\end{bmatrix} \in \mathbb{R}^{n\times m}
$$

So this means that our vectorized implementation becomes

$$
\begin{aligned}
&Z^{[1]}= W^{[1]}X + b^{[1]} \\
&A^{[1]} = \sigma(z^{[1]})\\
&Z^{[2]}= W^{[2]}A^{[1]} + b^{[2]}\\
&A^{[2]} = \sigma(Z^{[2]})\\
\end{aligned}
$$

with $Z^{[1]}$ and $A^{[1]}$ represent the $z$-values and $a$-values of the first layer of the neural network:

$$
Z^{[1]}=
\begin{bmatrix}
& \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & & \rule[-1ex]{0.5pt}{2.5ex}\\
&z^{[1](1)}&z^{[1](2)}&\dots&z^{[1](m)}\\
& \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & & \rule[-1ex]{0.5pt}{2.5ex}\\
\end{bmatrix} \in \mathbb{R}^{n^{[1]} \times m} 
\qquad 
A^{[1]}=
\begin{bmatrix}
& \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & & \rule[-1ex]{0.5pt}{2.5ex}\\
&a^{[1](1)}&a^{[1](2)}&\dots&a^{[1](m)}\\
& \rule[-1ex]{0.5pt}{2.5ex} & \rule[-1ex]{0.5pt}{2.5ex} & & \rule[-1ex]{0.5pt}{2.5ex}\\
\end{bmatrix} \in \mathbb{R}^{n^{[1]} \times m}
$$

where $m$ is the number of training examples and $n^{[1]}$ is the number of nodes (hidden units) in the first layer of the neural networks.
