---
layout: default
categories: linearRegression
title: "Linear Regression - Cost Function"
permalink: /ML3/
order: 3
comments: true
---

# Cost Function
We can measure the accuracy of our hypothesis function by using a cost function. The cost function calculates the **distance** of predicted data from observed data, obviously we want our predicted data to be as close as possible to truth values, in other words we want to **minimize** the distance from truth values.

This version of cost function takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from $x$ and the actual output $y$.

$$\begin{align}
J(\theta_0,\theta_1) & = \frac{1}{2m}\sum^m_{i=1}\left(\hat{y}_i-y_i\right)^2 \\
& = \frac{1}{2m}\sum^m_{i=1}\left(h_{\theta}(x_i) - y_i \right)^2 \\
& = \frac{1}{2m}\sum^m_{i=1}\left(\left(\theta_0 + \theta_{1}x^{(i)}\right) - y_i \right)^2 
\end{align}$$

To break it apart, it is $\frac{1}{2} \bar{x}$ where $\bar{x}$ is the mean of the squares of $h_\theta (x_{i}) - y_{i}$ or the difference (**distance**) between the predicted value and the actual value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $\left(\frac{1}{2}\right)$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term.

The idea is to chose $\theta_0, \theta_1$ to so that $h_\theta(x)$ is close to $y$ for each training example $(x,y)$. In other words we want to chose $\theta_0, \theta_1$ to minimize the cost function $J\left(\theta_0, \theta_1 \right)$

If we try to think of it in visual terms, our training data set is scattered on the $x,y$ plane. We are trying to make a straight line (defined by $h_\theta(x)$) which passes through these scattered data points. 

Our objective is to get the best possible line. The best possible line will that line for which the average squared vertical distance of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of $J(\theta_0, \theta_1)$ would be 0. The following example shows the ideal situation where we have a cost function of 0. 

For a simplified version of the regression hypothesis $h_\theta(x)$ where we removed the offset ($\theta_0$):

$$h_\theta(x)=\theta_1x$$


![png](ML-3-CostFunction_files/ML-3-CostFunction_8_0.png)


$$
\begin{align}
J(\theta_1) &= \frac{1}{2m}(0^2+0^2+0^2) \\
&= 0
\end{align}
$$

When $\theta_1 = 1$, we get a slope of 1 which goes through every single data point in our model. Conversely, when $\theta_1 = 0.5$, we see the vertical distance from our fit to the data points increase. 


![png](ML-3-CostFunction_files/ML-3-CostFunction_11_0.png)


Plotting 15 $\theta_1$ values in the interval $[-0.5, 2.5]$ yields a bell shaped graph 


![png](ML-3-CostFunction_files/ML-3-CostFunction_13_0.png)


Thus as a goal, we should try to minimize the cost function. In this case, $\theta_1 = 1$ is our global minimum. 

## Cost function visualization for two parameters
Let's take a slightly more complex hypothesis wwhere we have $\theta \in \mathbb{R}^2$. Suppose that we set $\theta_0=50; \;\theta_1=0.06$ amd that we want to plot the corresponding value for the cost function


![png](ML-3-CostFunction_files/ML-3-CostFunction_15_0.png)


Now we have two parameters so the cost function depends on two variables and its plot needs to account for three dimensions. We can use a surface plot where the $x$ axis is $\theta_0$, the $y$ axis is $\theta_1$ and the $z$ axis (the height) is the cost function at specific values of $\theta_0, \theta_1$


![png](ML-3-CostFunction_files/ML-3-CostFunction_17_0.png)


These surface plots allow to visualize the cost as a function of two parameters but they are somehow difficult to interpret and the perception of the shape of the cost-function-space is influenced by the perspective from a certain the point of view.

A more accessible kind of plots are the contour plots, bidimensional plot in which the two axis represent the parameters and lines join points where the $J(\theta_0, \theta_1)$ assumes the same value (like geograpical height-maps).


![png](ML-3-CostFunction_files/ML-3-CostFunction_19_0.png)

