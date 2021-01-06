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

---
The idea is to chose $\theta_0, \theta_1$ to so that $h_\theta(x)$ is close to $y$ for each training example $(x,y)$. In other words we want to chose $\theta_0, \theta_1$ to minimize the cost function $J\left(\theta_0, \theta_1 \right)$

---
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




    Text(0.5, 1.0, 'Cost function')




![png](ML-3-CostFunction_files/ML-3-CostFunction_15_1.png)



```python

```




    array([1.e+001, 1.e+002, 1.e+003, 1.e+004, 1.e+005, 1.e+006, 1.e+007,
           1.e+008, 1.e+009, 1.e+010, 1.e+011, 1.e+012, 1.e+013, 1.e+014,
           1.e+015, 1.e+016, 1.e+017, 1.e+018, 1.e+019, 1.e+020, 1.e+021,
           1.e+022, 1.e+023, 1.e+024, 1.e+025, 1.e+026, 1.e+027, 1.e+028,
           1.e+029, 1.e+030, 1.e+031, 1.e+032, 1.e+033, 1.e+034, 1.e+035,
           1.e+036, 1.e+037, 1.e+038, 1.e+039, 1.e+040, 1.e+041, 1.e+042,
           1.e+043, 1.e+044, 1.e+045, 1.e+046, 1.e+047, 1.e+048, 1.e+049,
           1.e+050, 1.e+051, 1.e+052, 1.e+053, 1.e+054, 1.e+055, 1.e+056,
           1.e+057, 1.e+058, 1.e+059, 1.e+060, 1.e+061, 1.e+062, 1.e+063,
           1.e+064, 1.e+065, 1.e+066, 1.e+067, 1.e+068, 1.e+069, 1.e+070,
           1.e+071, 1.e+072, 1.e+073, 1.e+074, 1.e+075, 1.e+076, 1.e+077,
           1.e+078, 1.e+079, 1.e+080, 1.e+081, 1.e+082, 1.e+083, 1.e+084,
           1.e+085, 1.e+086, 1.e+087, 1.e+088, 1.e+089, 1.e+090, 1.e+091,
           1.e+092, 1.e+093, 1.e+094, 1.e+095, 1.e+096, 1.e+097, 1.e+098,
           1.e+099, 1.e+100])




```python

```
