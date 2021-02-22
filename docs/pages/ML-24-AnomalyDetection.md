---
layout: default
title: "Anomaly Detection"
categories: anomaly
permalink: /ML24/
order: 24
comments: true
---

# Anomaly Detection
Anomaly detection describes a class of problems that many consider unsupervised but also have some aspect of supervised problems.

Anomaly detection is best explained through an example: suppose you are a aircraft engine manufacturer and, as part of your quality assurance testing, you measure a set of features of your manufactured engines. For this example the heat generated $x_1$ and the vibration intensity $x_2$.

The results of your measurements is a dataset $\left \lbrace x^{(1)}, x^{(2)}, \ldots, x^{(m)} \right \rbrace$ (<a href="#engines">Figure 25</a>)


    

<figure id="engines">
    <img src="{{site.baseurl}}/pages/ML-24-AnomalyDetection_files/ML-24-AnomalyDetection_3_0.png" alt="png">
    <figcaption>Figure 25. Results of quality assurance measurements performed on newly manufactured engines plotted in their features space $x_1, x_2$ and a new engine $x_\text{test}$</figcaption>
</figure>

Given a new example $x_\text{test}$, anomaly detection tries to answer the question: is the new example anomalous in any way?

In order to answer this question we are going to build a model of the probability of $x$ to be in a specific point of the feature space $p(x)$; If $p(x_\text{test}) < \varepsilon$ we will flag it as an anomaly (where $\varepsilon$ is a small number).

## Gaussian Distribution
Suppose $x \in \mathbb{R}$. If $x$ distributes as a Gaussian distributions with mean $\mu$ and variance $\sigma^2$, it means that its distributes resembles that in <a href="#gaussian">Figure 26</a>.


    

<figure id="gaussian">
    <img src="{{site.baseurl}}/pages/ML-24-AnomalyDetection_files/ML-24-AnomalyDetection_6_0.png" alt="png">
    <figcaption>Figure 26. Gaussian distribution with mean $\mu$ and variance $\sigma^2$</figcaption>
</figure>

The function of the Gaussian distribution is

$$
p(x;\mu, \sigma^2)=\frac{1}{\sigma \sqrt{2 \pi}}  \left( - \frac{(x-\mu)^2}{2\sigma^2} \right)
$$

## Parameter estimation
Suppose you have a dataset $\lbrace x^{(1)}, x^{(2)}, \ldots, x^{(m)} \rbrace$ with $x^{(i)} \in \mathbb{R}$ (<a href="#paramestim">Figure 27</a>) and you suspect that they are Gaussian distributed with each $x^{(i)} \approx \mathcal{N}(\mu, \sigma^2)$ but I don't know the values of the two parameters $\mu$ and $\sigma^2$


    

<figure id="paramestim">
    <img src="{{site.baseurl}}/pages/ML-24-AnomalyDetection_files/ML-24-AnomalyDetection_10_0.png" alt="png">
    <figcaption>Figure 27. Data distributed on the $x$ axis and their Gaussian density estimation</figcaption>
</figure>

The parameters can be estimated from $x$, we will have

$$
\mu= \frac{1}{m} \sum^m_{i=1}x^{(i)}  \qquad \qquad \sigma^2 = \frac{1}{m}\sum^m_{i=1}(x^{(i)} - \mu)^2
$$

## Anomaly detection algorithm
Given a $m \times n$ training set $x \in \mathbb{R}^n$

The anomaly detection algorithm requires that parameters are fitted **for each feature $x_j$**:

$$
\mu_j= \frac{1}{m} \sum^m_{i=1}x_j^{(i)}  \qquad \qquad \sigma_j^2 = \frac{1}{m}\sum^m_{i=1}(x_j^{(i)} - \mu_j)^2
$$

Then, Given a  new example $x$, compute $p(x)$:

$$
\begin{align}
p(x) & = p(x_1; \mu_1,\sigma^2_1)p(x_2; \mu_2,\sigma^2_2),\ldots,p(x_n; \mu_n,\sigma^2_n) \\
& = \prod^n_{j=1}p(x_j; \mu_j,\sigma^2_j) \\
& = \prod^n_{j=1}\frac{1}{\sigma_j \sqrt{2 \pi}}  \left( - \frac{(x_j-\mu_j)^2}{2\sigma_j^2} \right)
\end{align}
$$

### Evaluation
Anomaly detection algorithm can be evaluated with a confusion matrix given that some anomalous data is included in the training, in the test and cross-validation sets. The number of anomalous examples can be very small (e.g. 1:1000) but it is useful when evaluating.

Evaluation on test / cross-validation set can the be based on the raw numbers of the confusion matrix, on precision/recall or $F_1$-Score metrics.

The cross-validations set can be used to choose the parameter $\varepsilon$

## Anomaly Detection vs Supervised Learning
In order to evaluate an anomaly detection model we need some labeled data. Then why don't we use a supervised classification algorithm to detect anomalous data points?

|           | Anomaly detection                                                                                                                                                                                                                                | Supervised learning                                                                                                                                                                      |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Imbalance | Commonly, you have a very small number of positive examples (0-20) and  a large number of negative examples.                                                                                                                                     | You have a large number of positive and negative examples                                                                                                                                |
| Coverage  | There may be many different types of anomalies. It may be hard for any algorithm to learn from positive examples what do the anomalies look like. Furthermore, future anomalies may look nothing like any anomalous example in the training set. | You have enough positive examples for an algorithm to get a sense of what positive examples are like; future positive examples are likely to be similar to the ones in the training set. |

## Selecting Features for anomaly detection
The choice of features included in an anomaly detection model heavily impact its performance.

The most common problem that we want to overcome and is caused by a sub-optimal choice of the features is that $p(x)$ assumes comparable values for normal and anomalous examples.

Usually the best procedure to choose features is by an error analysis, which is best explained in the <a href="#erroranalysis">Figure 28</a>


    

<figure id="erroranalysis">
    <img src="{{site.baseurl}}/pages/ML-24-AnomalyDetection_files/ML-24-AnomalyDetection_14_0.png" alt="png">
    <figcaption>Figure 28. An anomalous example not detected with a single feature $x_1$ and correctly detected in the feature space $x_1, x_2$</figcaption>
</figure>

A good practice in choosing features is to choose or combine features so that they take on unusually large or small values in the event of an anomaly.

## Non-Gaussian features
Even if a feature has not a Gaussian distribution (<a href="#gaussbetadist">Figure 29</a>), usually the algorithm works fine. However, usually the algorithm will work better if non-Gaussian data is **transformed to Gaussian**.


    

<figure id="gaussbetadist">
    <img src="{{site.baseurl}}/pages/ML-24-AnomalyDetection_files/ML-24-AnomalyDetection_16_0.png" alt="png">
    <figcaption>Figure 29. Histogram of data drawn from a Gaussian (A) and Beta (B) distributions and from the Beta distribution transformed to Gaussian (C)</figcaption>
</figure>
