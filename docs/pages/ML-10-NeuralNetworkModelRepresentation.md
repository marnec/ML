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

![neuron](data/img/neuron.png)

In a simplistic way a neuron is a computational unit that receive some input via dendrites, does some computation and then outputs something via the axon to other neurons in the brain.

A neuron implemented on the computer has a very simple model that mimics the architecture of an anatomical neuron. We're a going to model a neuron as just a logistic unit. The yellow node represents the *body* of the neuron, which is fed input through its *dendrites*, and produces an output $h_\theta(x)$ that is produced by the neuron body, though its **activation function** and transported forward by the neuron *axon*. Where $h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}$


    
![png](ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_2_0.png)
    


A simpler representation is sometimes used to depict a neural network

$$
[x_0x_1x_2x_3]\to[]\to h_\theta(x)
$$

Sometimes when representing the inputs of a neuron the $x_0$, which is sometimes called the bias unit (since $x_0=1$), might be added if it convenient for the discussion of the model.


    
![png](ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_4_0.png)
    


In neural networks the parameters $\theta$ of the model are sometimes called **weights** ($w$).

Until now we represented single neurons; a neural network is a group of different neurons connected together. The input nodes are grouped in what is called the **input layer** ($a^{[0]}$), which is always the first layer of the neural network. The final layer is called the **output layer**, since it computes the final value of our hypothesis. And all layers in between the input and the output layers are called **hidden layers**. They are called hidden layers because we can't observes the values computed by these nodes.


    
![png](ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_6_0.png)
    


<i id="simpleann">A simple neural network with one hidden layer</i>

The computational entities in a neural networks are:

* $a_i^{[j]}$  activation neuron/unit $i$ in layer $j$
* $w^{(j)}$ matrix of weights controlling the function mapping from layer $j$ to layer $j+1$ 

And the computation in the network

$$
\left[x_0 x_1 x_2 x_3 \right]\to \left[a_1^{[1]}a_2^{[1]}a_3^{[1]} \right]\to h_w(x) \equiv a_1^{[2]} \equiv \hat{y}
$$

depicted in the figure above, is

$$
\begin{align}
& a_1^{[1]} = g \left(w_{10}^{[1]}x_0 + w_{11}^{[1]}x_1 + w_{12}^{[1]}x_2 + w_{13}^{[1]}x_3\right) \\
& a_2^{[1]} = g \left(w_{20}^{[1]}x_0 + w_{21}^{[1]}x_1 + w_{22}^{[1]}x_2 + w_{23}^{[1]}x_3\right) \\
& a_3^{[1]} = g \left(w_{30}^{[1]}x_0 + w_{31}^{[1]}x_1 + w_{32}^{[1]}x_2 + w_{33}^{[1]}x_3\right)
\end{align}
\label{eq:neuralnet} \tag{1}
$$


$$
h_w(x)= a_1^{[3]} =  \left(w_{10}^{[2]}a_0^{[2]} + w_{11}^{[1]}a_1^{[2]} + w_{12}^{[2]}a_2^{[2]} + w_{13}^{[2]}a_3^{[2]}\right)
\label{eq:neuralnet_h} \tag{2}
$$

So in this network we have 3 input units and 3 hidden units and so the dimension of $w^{[1]}$, which is the matrix of parameters weighting the values from the 3 input units from the 3 hidden units is going to be $w^{[1]} \in \mathbb{R} ^{3\times4}$.

In general if a network has $s_j$ units in layer $j$, $s_{j+1}$ units in layer $j+1$, then $w^{[j]}$ will be of dimension $s_{j+1} \times (s_j+1)$

# Forward propagation

Let's rewrite the argument of the functions $g$ in $\eqref{eq:neuralnet}$ as $z^{[j]}$ so that now we have

$$
\begin{align}
& a_1^{[1]} = g \left(z_1^{[1]}\right) \\
& a_2^{[1]} = g \left(z_2^{[1]}\right) \\
& a_3^{[1]} = g \left(z_3^{[1]}\right)
\end{align}
$$

Looking at $\eqref{eq:neuralnet}$ again, we can see that the way the arguments of $g$ in $a_1^{[1]}, a_2^{[1]}, a_2^{[1]}$ are disposed can be written as $w^{[1]}x$, where $x$ is a vector of inputs.

$$
\begin{split}
x=
\begin{bmatrix}
x_0\\
x_1\\
x_2\\
x_3
\end{bmatrix}
\end{split}
\quad\quad\quad
\begin{split}
z^{[1]}=
\begin{bmatrix}
z_1^{[1]}\\
z_2^{[1]}\\
z_3^{[1]}
\end{bmatrix}
\end{split}
$$

$$
\begin{align}
&z^{[1]}=w^{[1]}x \\
&a^{[1]}=g\left(z^{[1]}\right)
\end{align}
$$

Where $a^{[1]}$ and $z^{[1]}$ are $\mathbb{R}^3$ vectors. Now we could say that the input layer is also an activation layer and call it $a^{[0]}$ so that

$$
z^{[1])}=w^{[1]}a^{[0]}
$$

What we have written so far give us the value for $ a^{[1]}_1, a^{[1]}_2, a^{[1]}_3 $.

If we look at $ \eqref{eq:neuralnet_h} $ we see that we need one more value, the bias unit $a^{[1]}_{0} = 1$ that we need to add to $a^{[1]}$, which becomes a $\mathbb{R}^4$ vector

$$
a^{[1]}=
\begin{bmatrix}
a_0^{[1]}\\
a_1^{[1]}\\
a_2^{[1]}\\
a_3^{[1]}
\end{bmatrix}
$$

We can now compute $h_w(x)$ $\eqref{eq:neuralnet_h}$

$$
\begin{align}
&z^{[2]}=w^{[2]}a^{[1]} \\
&h_w(x)=a^{[2]} = g\left(z^{[2]}\right)
\end{align}
$$

or more generally

$$
\begin{align}
&z^{[j]}=\Theta^{[j]}a^{[j-1]} \\
&h_w(x)=a^{[j]} = g\left(z^{[j]}\right)
\end{align}
$$


This process is called **forward propagation**

# Neural networks learn its own features
Let's take the network used as example above and focus on the last two layers


    
![png](ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_10_0.png)
    


What is left in this neural network is simply logistic regeression, where we use the output unit (or logistic regression unit) to build the hypothesis $h_w(x)$

$$
h_w(x) = g\left(w_{10}^{[2]}a_0^{[1]}+w_{11}^{[2]}a_1^{[1]}+w_{12}^{[2]}a_2^{[1]}+ w_{13}^{[2]}a_3^{[1]} \right)
$$

Where the features fed into logistic regression are the values in $a^{[1]}$. And here resides the fundamental difference between neural networks and logistic regression: the features $a^{[1]}$ they themselves are learned as functions of the input $x$ with some other set of parameters $w^{[1]}$

The neural network, instead of being constrained to feed the features $x$ to logistic regression, learns its own features $a^{[1]}$ to feed into logistic regression. Depending on the parameters $w^{[1]}$, it can learn some complex features and result in a better hypothesis that you could have if you were constrained to use features $x$ or even if you had to manually set some higher order polynomial features combining the features $x$.

Neural networks can have different number and dimension of hidden layers and the way a neural network is connected is called its **architecture**.


    
![png](ML-10-NeuralNetworkModelRepresentation_files/ML-10-NeuralNetworkModelRepresentation_12_0.png)
    


## Vectorization
<a href="#simpleann">The simple one-hidden-layer neural network depicted above</a> is described by the set of equations:

$$
\begin{aligned}
&z^{[1]}= W^{[1]}x + b^{[1]} \\
&a^{[1]} = \sigma(z^{[1]})\\
&z^{[2]}= W^{[1]}a^{[1]} + b^{[2]}\\
&a^{[2]} = \sigma(z^{[2]})\\
\end{aligned}
$$

This process must be repeated for each training example $x^{(n)}$ and will produce $n$ outputs $a^{[2](n)} = \hat{y}^{(n)}$

In a non-vectorized implementation you would have something along the lines of:

```python
for i in len(examples):
    z[1][i] = w[1] @ x(i) + b[i]
    a[1][i] = sigmoid(z[1][i])
    z[2][i] = w[2] @ x(i) + b[i]
    a[2][i] = sigmoid(z[2][i])
```

Given our vector of training examples:

$$
X=
\begin{bmatrix}
&\vdots&\vdots&&\vdots\\
&x^{(1)}&x^{(2)}&\dots&x^{(m)}\\
&\vdots&\vdots&&\vdots\\
\end{bmatrix} \in \mathbb{R}^{n\times m}
$$

So this means that our vectorized implementation becomes

$$
\begin{aligned}
&Z^{[1]}= W^{[1]}X + b^{[1]} \\
&A^{[1]} = \sigma(z^{[1]})\\
&Z^{[2]}= W^{[1]}A^{[1]} + b^{[2]}\\
&A^{[2]} = \sigma(Z^{[2]})\\
\end{aligned}
$$

with $Z^{[1]}$ and $A^{[1]}$ represent the $z$-values and $a$-values of the first layer of the neural network:

$$
Z^{[1]}=
\begin{bmatrix}
&\vdots&\vdots&&\vdots\\
&z^{[1](1)}&z^{[1](2)}&\dots&z^{[1](m)}\\
&\vdots&\vdots&&\vdots\\
\end{bmatrix} \in \mathbb{R}^{n^{[1]} \times m} 
\qquad 
A^{[1]}=
\begin{bmatrix}
&\vdots&\vdots&&\vdots\\
&a^{[1](1)}&a^{[1](2)}&\dots&a^{[1](m)}\\
&\vdots&\vdots&&\vdots\\
\end{bmatrix} \in \mathbb{R}^{n^{[1]} \times m}
$$

where $m$ is the number of training examples and $n^{[1]}$ is the number of nodes (hidden units) in the first layer of the neural networks.
