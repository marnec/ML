---
layout: default
title: "Neural Networks - Backpropagation"
categories: neuralNetwork
permalink: /ML13/
order: 13
comments: true
---

# Gradien Descent
The aim of a neural network is, to minimize the cost function calculated on its parameters, which, if using the same network as in 
<a href="ML10#simpleann">ML-10</a>, are: 

$$
\begin{align}
& W^{[1]} \in \mathbb{R}^{(n^{[1]}, n^{[0]})} \\
& b^{[1]} \in \mathbb{R}^{(n^{[1]}, 1)} \\
& W^{[2]} \in \mathbb{R}^{(n^{[2]}, n^{[1]})} \\
& b^{[2]} \in \mathbb{R}^{(n^{[2]}, 1)}
\end{align}
$$

The cost function ($J$) is defined as the average over the training examples of the loss function ($\mathcal{L}$) for a single example:

$$
J(W^{[1]},b^{[1]}, W^{[2]}, b^{[2]}) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}\left(\hat{y}, y \right)
$$

where, if the network is used for binary classification, $\mathcal{L}$ can be exactly the same as in logistic regression.

In order to train the network, we will need to perform gradient descent, so:

* For each iteration until convergence:

    1. compute the prediciton $\hat{y}$
    2. compute the derivatives $\frac{\partial J}{\partial W^{[1]}}, \frac{\partial J}{\partial b^{[1]}},\frac{\partial J}{\partial W^{[2]}}, \frac{\partial J}{\partial b^{[2]}}$
    3. update the parameters $W^{[1]}=W^{[1]}-\alpha \frac{\partial J}{\partial W^{[1]}},\cdots$


# Backpropagation
In order to compute the derivatives in a neural network, we use a technique called **backpropagation**, where we proceed step by step backwards in the computation of the contribution to $J$ from weights of the different layers from the rightmost to the leftmost taking advantage of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).

The derivative $dW^{[2]}= \frac{\partial J}{\partial W^{[2]}}$ and $db^{[2]}= \frac{\partial J}{\partial b^{[2]}}$ are esaily calculated in two steps:

$$
\begin{aligned}
&dZ^{[2]}=A^{[2]} - Y\\
&dW^{[2]}= \frac{1}{m}dZ^{[2]}A^{[1]T}\\
&db^{[2]}= \frac{1}{m}\sum dZ^{[2]}
\end{aligned}
$$

We can see that in order to calculate $dW^{[2]}$ and $db^{[2]}$, we need to calculate the term $dZ^{[2]}$, that represents the error a layer.

To calculate $dW^{[1]}$ and $db^{[1]}$, we need $dZ^{[2]}$, which is calculated building on the the calculation in the previous layer

$$
\begin{aligned}
&dZ^{[1]}= W^{[2]T} dZ^{[2]} \odot g' (Z^{[1]})\\
&dW^{[1]}= \frac{1}{m}dZ^{[2]}\\
&db^{[1]}= \frac{1}{m}\sum dZ^{[1]}
\end{aligned}
$$


```python
dot = Digraph(node_attr={'fontsize':'9'}, edge_attr={'arrowsize': '0.5', 'fontsize':'9'}, engine='dot')
dot.attr(rankdir='LR', packmode='graph')

with dot.subgraph() as sg:
    sg.attr(rank='equal')
    sg.node('x', shape='plaintext', margin='0')
    sg.node('w', label='W[1]', shape='plaintext', margin='0')
    sg.node('b', label='b[1]', shape='plaintext', margin='0')


    
dot.node('z', shape='rect', label='z[1] = W[1] x + b[1]', margin='0')

dot.node('h', shape='rect', label='z[2] = W2 a[1] + b[2]', margin='0')
dot.node('y', shape='rect', label='a[2] = g(z[2])', margin='0')
dot.node('l', shape='rect', label='L(a[2], y)', margin='0')

with dot.subgraph() as sg:
    sg.node('j', label='W[2]', shape='plaintext', margin='0')
    sg.node('k', label='b[2]', shape='plaintext', margin='0')
    sg.node('a', shape='rect', label='a[1] = g(z[1])', margin='0')
    
dot.edges(['xz', 'wz', 'bz', 'za', 'ah', 'hy', 'yl', 'jh', 'kh'])
dot.edge('l', 'y', headport='s', tailport='s', color='red', label='da[2]', fontcolor='red')
dot.edge('y', 'h', headport='s', tailport='s', color='red', label='dz[2]', fontcolor='red')
dot.edge('h', 'j', headport='s', tailport='s', color='red', label='dz[2]', fontcolor='red')
dot.edge('h', 'j', color='red', label='dz[2]', fontcolor='red')
dot
```




    
![svg](ML-13-NeuralNetworkOptimization_files/ML-13-NeuralNetworkOptimization_2_0.svg)
    



# The following section is deprecated and will be dropped 
### Gradient for a single example
Suppose that we have only one training example $(x, y)$ and the neural network in the picture below


    
![png](ML-13-NeuralNetworkOptimization_files/ML-13-NeuralNetworkOptimization_4_0.png)
    


The first thing we are going to do is applying **forward propagation**

$$
\begin{align}
&a^{(1)} = x \\ 
&a^{(2)} = g\left(\Theta^{(1)}a^{(1)}\right); \quad (\text{add }  a_0^{(2)})\\ 
&a^{(3)} = g\left(\Theta^{(2)}a^{(2)}\right); \quad (\text{add }  a_0^{(3)})\\ 
h_\Theta(x) =\; &a^{(4)} = g\left(\Theta^{(3)}a^{(3)}\right)\\ 
\end{align}
$$

In order to compute the derivative we are going to use an algorithm called backpropagation. For each node we are going to compute the term $\delta_j^{(l)}$ that will represent the error of node $j$ in layer $l$ with resepect to $a_j^{(l)}$. For the neural network in the picture we are going to compute $\delta_j^{(l)}$ for each output unit (layer $L=4$) and then proceed backward and compute the error for the previous layers (hence the name back-propagation).

$$
\begin{align}
&\delta^{(4)} = a^{(4)} - y \\
&\delta^{(3)} = \left(\Theta^{(3)}\right)^T\delta^{(4)} \; \odot \; g'\left(z^{(3)}\right)\\ 
&\delta^{(2)} = \left(\Theta^{(2)}\right)^T\delta^{(3)} \; \odot \; g'\left(z^{(2)}\right)\\  
\end{align}
$$

Where

* $\delta^{(l)}$ and $a^{(l)}$ are the vectors of respectively errors and activation values at layer $l$; 
* $\odot$ is the symbol for element-wise (Hadamard) product;
* $g'\left(z^{(l)}\right)$ is the derivative of the activation function $g$ at the input value given by $z^{(l)}$ and is computed as $a^{(l)} \odot \left(1-a^{(l)}\right)$

There is no $\delta^{(1)}$ term because layer 1 is the input layer and contains the observed features so there is no error associated to that (there might be an error but we don't want to change those values so we don't want to define a $\delta^{(l)}$ error).

Finally we want to compute $\eqref{eq:partdev}$. The derivation becomes quite complicated but it is possible to prove that, ignoring regularization or setting the regularization term $\lambda=0$

$$
\begin{equation}
\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta) = a_j^{(l)}\delta_i^{(l+i)} \quad \quad (\text{ignoring }\lambda; \text{ if } \lambda=0 )
\end{equation}
\label{eq:deriv} \tag{3}
$$

### Gradient for multiple examples
Now we can extend the case discussed for one example to multiple examples $\lbrace \left( x^{(1)}, y^{(1)}\right), \cdots, \left( x^{(m)}, y^{(m)}\right) \rbrace$

The first step will be to set $\Delta_{ij}^{(l)} = 0$ that will accumulate all the errors and will be used to compute $\frac{\partial}{\partial\Theta_{ij}^{()}l}J(\Theta)$

Then we are going to loop through each example from $i=1$ to $m$, and for each example $\left( x^{(i)}, y^{(i)}\right)$ we will

1. Set $a^{(1)} = x^{(i)}$
2. Perform forward propagation to compute $a^{(l)}$ for $l = 2, 3, \dots, L$
3. Use the output label for the current example $y^{(i)}$, to compute the error term $\delta^{(L)}=\Delta_{ij}^{(l)} = 0 - y^{(i)}$
4. Back-propagate the error down to the first hidden layer $\delta^{(L-1)}, \delta^{(L-2)}, \dots \delta^{(2)}$
5. Accumulate the partial derivatives $\Delta_{ij}^{(l)} := \Delta_{ij}^{(l)} + a_j^{(l)}\delta_i^{(l+1)}$ (or in its vectorized form $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}\left(a^{(l)}\right)^T$)

Now we can finally calculate the accumulator term $D_{ij}^{(l)}$, the calculation of which can take two forms depending on $j$:

$$
\begin{align}
D_{ij}^{(l)} := \frac{1}{m}\Delta_{ij}^{(l)} + \lambda\Theta_{ij}^{(l)} &\quad\quad \text{if } j \neq 0 \\
D_{ij}^{(l)} := \frac{1}{m}\Delta_{ij}^{(l)} &\quad\quad \text{if } j = 0
\end{align}
$$

Where the case of $j=0$ corresponds to the bias term and that is why we don't regularize in that case.

It can then be demonstrated that $D_{ij}^{(l)}$ is equal to the partial derivative of $J(\Theta)$ respect to $\Theta$

$$
\frac{\partial}{\partial\Theta}J(\Theta)=D_{ij}^{(l)}
$$

## Back-propagation inuition
To try and understand back-porpagation let's first see exactly what is happening in forward propagation


    
![png](ML-13-NeuralNetworkOptimization_files/ML-13-NeuralNetworkOptimization_8_0.png)
    


Let's take the network depicted above, the count of units (excluding the bias) are 2 for the input layer and for the two input layers and 1 for the output layer 


    
![png](ML-13-NeuralNetworkOptimization_files/ML-13-NeuralNetworkOptimization_10_0.png)
    


When performing forward propagation for one example $x^{(i)}, y^{(i)}$, we will feed $x^{(i)}$ in the input layer ($x_1^{(i)}, x_2^{(i)}$). The computation of $z^{(3)}$  in forward propagation is:

$$
\begin{equation}
z^{(3)}
=\color{magenta}{\Theta_{10}^{(2)} \cdot 1}
+\color{red}{\Theta_{11}^{(2)}a_1^{(2)}}
+\color{cyan}{\Theta_{12}^{(2)}a_2^{(2)}}
\end{equation}
\label{eq:forprop} \tag{4}
$$

Back propagation does something very similar to $\eqref{eq:forprop}$ except that the direction of the operation is reversed.

To understand what back-propagation is doing let's focus on the cost function $\eqref{eq:neuralnetcost}$. Since we have just one output unit we can simplify $\eqref{eq:neuralnetcost}$ to 

$$
\begin{align}
J(\Theta)=&-\frac{1}{m}\left[\sum^m_{i=1}y^{(i)}\log \left(h_\Theta\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right)\log\left(1-h_\Theta \left(x^{(i)}\right)\right)\right] \\
&+\frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}}\left(\Theta_{ji}^{(l)}\right)^2
\end{align}
$$

Now since forward and back-propagation are applied to one example at a time, let's focus on a single example $x^{(i)}, y^{(i)}$  and ignore regularization ($\lambda=0$); the cost function $\text{cost}$

$$
\begin{equation}
\text{cost}(i) = y^{(i)}\log \left(h_\Theta\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right)\log\left(1-h_\Theta \left(x^{(i)}\right)\right)
\end{equation}
\label{eq:costi} \tag{5}
$$

calculates the distance between our prediction $h_\Theta(x)$ and labels $y$ in the case of a logistic function. In a way this is very similar to the cost function for linear regression and for simplicity we may think the cost function as

$$\text{cost}(i) \approx \left(h_\Theta\left(x^{(i)}\right) - y^{(i)}\right)^2$$

Now let's look at what back-propagation is doing

In a previous section we said that back-propagation computes $\delta_j^{(l)}$ and called that term the "error" $a_j^{(l)}$. More formally $\delta_j^{(l)}$ are the partial deriative with respect to the intermediate terms $z_j^{(l)}$ of the cost function $\eqref{eq:costi}$.

$$\delta_j^{(l)} = \frac{\partial}{\partial z_j^{(l)}}\text{cost}(i)$$

And they are a measuer of how much we would like to change the neural networks weight in order to affect the intermediate terms $z_j^{(l)}$


    
![png](ML-13-NeuralNetworkOptimization_files/ML-13-NeuralNetworkOptimization_13_0.png)
    


$$
\delta_2^{(2)}=\color{magenta}{\Theta_{12}^{(i)}\delta_1^{(3)}}+\color{red}{\Theta_{22}^{(3)}\delta_2^{(3)}}
$$
