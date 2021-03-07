---
layout: default
title: "Logistic Regression - Model Representation"
categories: logisticRegression
permalink: /ML7/
order: 7
comments: true
---

# Classification problems and Logistic regression

Classification problems are a type of supervised learning where input variables are mapped to a predicted value $(y)$, which assumes discrete values. In the simplest case:

$$y \in \{0,1\}$$

Where $0$ can sometimes be called the "Negative class" and $1$ the "Positive class". The designation of $0$ being negative and $1$ positive is arbitrary. And this arbitrary associations reflects the intuition that $1$ means presence of something while $0$ means absence of something.

In multiclass classification problems $y \in \{0,1,\dots,n\}$ but we will start from binary classification problems

Let's take an example where we have to predict if a tumor is malignant ($y=1$) or bening ($y=0$) from the tumor size. Let's say that we have this situation.


    
![png](ML-7-Classification_files/ML-7-Classification_3_0.png)
    


If we use linear regression here we might have something like this


    
![png](ML-7-Classification_files/ML-7-Classification_5_0.png)
    


And it seems reasonable because we can imagine to apply a threshold to $y$ at $0.5$ so that:

* If $h_\theta(x) \geq  0.5$ the classifier predicts $y=1$
* If $h_\theta(x) < 0.5$ the classifier predicts $y=0$

However if we add another Tumor size that is much bigger than the previous, $h_\theta(x)$ will also change


    
![png](ML-7-Classification_files/ML-7-Classification_7_0.png)
    


If we apply the 0.5 threshold now the we we will classify incorrectly two malignant tumors. When using linear regression for classification problems we might get lucky but in general it is not a good idea. We should instead use logistic regression.

It may be confusing that the term *regression* in logistic regression is associated with classification while linear *regression* is not, but that is just the name that it was given to it for historical reasons.

Logistic regression is a classification algorithm

# Hypothesis representation
The representation of the logistic regression hypothesis should satisfy the output boundaries

$$0 \leq H_\theta(x) \leq 1$$

For linear regression the hypothesis looks like

$$
\begin{equation*}
h_\theta(x) = \theta^Tx
\label{eq:linreg} \tag{1}
\end{equation*}
$$


For logistic regression we will modify $\eqref{eq:linreg}$ to

$$
\begin{equation*}
h_\theta(x) = g\left(\theta^Tx\right)
\label{eq:logreg_compact} \tag{2}
\end{equation*}
$$

Where

$$
\begin{equation*}
g(z)=\frac{1}{1+e^{-z}}
\label{eq:logfunc} \tag{3}
\end{equation*}
$$

$\eqref{eq:logfunc}$ is called Sigmoid function or Logistic function (they are synonyms) and this is the reason for the name of the algorithm (*logistic* regression).

Putting together $\eqref{eq:logreg_compact}$ and $\eqref{eq:logfunc}$ we have

$$
\begin{equation*}
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
\label{eq:logreg_full} \tag{4}
\end{equation*}
$$

The output $(y)$ of the sigmoid function

$$\begin{align}
&\lim_{z\to+\infty}g(z) = 1 \\
&\lim_{z\to-\infty}g(z) = 0 \\
&g(0) = \frac{1}{2}
\end{align}$$


    
![png](ML-7-Classification_files/ML-7-Classification_11_0.png)
    


And so we predict:

$$
\begin{equation*}
y=
\begin{cases}
1, & \text{if } h_\theta(x) \geq 0.5\\
0, & \text{otherwise}
\end{cases}
\label{eq:interpret} \tag{5}
\end{equation*}
$$
 
## Logistic regression output interpretation
The interpretation of the output of the logistic regression hypotheis output is that $y$ represent the probability of the class being positive given a certain $x$ parametrized by $\theta$.

$$
\begin{equation*}
h_\theta(x)= P(y=1|x;\theta)
\label{eq:conditional_prob} \tag{6}
\end{equation*}
$$

Since $y \in \{0,1\}$

$$
\begin{align}
&P(y=0|x;\theta) + P(y=1|x;\theta) = 1 \\
&P(y=0|x;\theta) = 1-P(y=1|x;\theta)
\end{align}
$$

# Decision Boundary
We know $\eqref{eq:interpret}$, now we want to know exactly when $h_\theta(x) \geq 0.5$. Looking at the plot of the sigmoid function we can see that

$$
\begin{equation*}
h_\theta(x) = g(z)
\begin{cases}
\geq 0.5, & \text{if } z \geq 0\\
<0.5, & \text{otherwise }\\
\end{cases}
\end{equation*}
$$

And since $z\equiv\theta^Tx$

$$
\begin{equation*}
h_\theta(x) = g(\theta^Tx)
\begin{cases}
\geq 0.5, & \text{if } \theta^Tx \geq 0\\
<0.5, & \text{otherwise }\\
\end{cases}
\label{eq:cond} \tag{7}
\end{equation*}
$$

This means that if we decide to adopt $0.5$ as a threshold as in $\eqref{eq:interpret}$, if we want to determine if $y=1$ we have to determine if $\theta^Tx \geq 0$


    
![png](ML-7-Classification_files/ML-7-Classification_13_0.png)
    


Let's suppose we have a dataset like that on the figure below


    
![png](ML-7-Classification_files/ML-7-Classification_15_0.png)
    


let's supppose that our hypothesis for this dataset is: 

$$
\begin{equation*}
h_\theta(x)=g(\theta_0+\theta_1x_1+\theta_2x_2)
\end{equation*}
\label{eq:loghypothesis} \tag{8}
$$

and that we chose some values for the parameters $\theta_J$ such as the paramter vector $\theta$ is

$$
\theta=
\begin{bmatrix}
3 \\
1 \\
1
\end{bmatrix}
\label{eq:thetat} \tag{9}
$$

Given $\eqref{eq:cond}$, and since $\theta^Tx=-3 +x_1 + x_2$

$$
\begin{equation*}
y=
\begin{cases}
1, & \text{if } -3 +x_1 + x_2 \geq 0\\
0, & \text{otherwise }\\
\end{cases}
\end{equation*}
$$

So for any couple of values $x_1, x_2$, we have $y=1$ whenever $x_1+x_2\geq 3$. This means that we can define a boundary region, at exactly $x_1+x_2=3$, that separates the $X_1,X_2$ space that corresponds to $y=1$ from the space that corresponds to $y=0$ for $\theta$ defined in $\eqref{eq:thetat}$.


    
![png](ML-7-Classification_files/ML-7-Classification_17_0.png)
    


This line (in this case $x_1+x_2=3$) is called the **decision boundary** and if defines the points for which $h_\theta(x)=0.5$. The decision boundary is a property of the hypothesis and is independent of the training set ($X$).

Now let's examine a more difficult case and discuss how we could define an hypothesis that   separates the $y=1$ space from the $y=0$ space when our dataset looks like this.


    
![png](ML-7-Classification_files/ML-7-Classification_19_0.png)
    


As for linear regression also in logistic regression we could add higher order polynomial terms to the features. In this case we could take $\eqref{eq:loghypothesis}$ and add polynomial terms

$$
\begin{align}
h_\theta(x)=g(\theta_0&+\theta_1x_1+\theta_2x_2\\
&+\theta_3x^2_1+\theta_4x^2_2)
\end{align}
$$

Now we have 5 paramters and lets suppose that we chose those parameters such as the vector $\theta$

$$
\theta^T=
\begin{bmatrix}
-1 \\
0 \\
0 \\
1 \\
1
\end{bmatrix}
\label{eq:tetatt} \tag{10}
$$

and $\theta^Tx=-1+x_1^2+x_2^2$. 

So $y=1$ for $x_1^2+x_2^2\geq1$ and the decision boundary is $x_1^2+x_2^2=1$.


    
![png](ML-7-Classification_files/ML-7-Classification_21_0.png)
    

