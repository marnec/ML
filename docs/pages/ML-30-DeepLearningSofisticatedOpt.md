---
layout: default
title: "Deep Learning - Speed up learning - Other optimization techinques"
categories: deeplearning
permalink: /ML30/
order: 30
comments: true
---

# More sophisticated optimization techniques
There are other more sophisticated optimization techniques than gradient descent, in order to talk about them let's introduce **exponentially weighted averages**

## Exponentially weighted averages
Let's take the average daily temperature in London across the year 2019 (<a href="#fig:londtemp">Figure 55</a>)


    

<figure id="fig:londtemp">
    <img src="{{site.baseurl}}/pages/ML-30-DeepLearningSofisticatedOpt_files/ML-30-DeepLearningSofisticatedOpt_2_0.png" alt="png">
    <figcaption>Figure 55. Average daily temperatures in the year 2019 in London</figcaption>
</figure>

The data looks noisy and in order to compute the trend of the temperature across the year we can use the following approach. Starting from the first day $v_0=0$ we can proceed by averaging:

$$
\begin{aligned}
&v_0=0\\
&v_1 = 0.9v_0 + 0.1 \theta_1 \\
&v_2 = 0.9v_1 + 0.1 \theta_2 \\
&v_3 = 0.9v_3 + 0.1 \theta_3 \\
& \vdots \\
&v_t = 0.9v_{t-1} + 0.1 \theta_t \\
\end{aligned}
$$

where $\theta_i$ is current temperature.

Let's rewrite the generalization as:

$$
\begin{equation}
v_t = \beta v_{t-1} + (1-\beta) \theta_t
\end{equation}
\label{eq:ewa} \tag{1}
$$

We can think of $v_t$ as approximately averaging over $\frac{1}{1-\beta}$ days, so when using:

* $\beta = 0.9 \to \frac{1}{1-0.9} \approx$ average 10 over days
* $\beta = 0.98 \to \frac{1}{1-0.98} \approx$ average over 50 days
* $\beta = 0.5 \to \frac{1}{1-5} \approx$ average over 2 days


    

<figure id="fig:ewa">
    <img src="{{site.baseurl}}/pages/ML-30-DeepLearningSofisticatedOpt_files/ML-30-DeepLearningSofisticatedOpt_4_0.png" alt="png">
    <figcaption>Figure 56. Exponentially weighted average applied to raw data with different values of $\beta$</figcaption>
</figure>

As we can see in <a href="#fig:ewa">Figure 56</a>, increasing values of $\beta$ will produce smoother trends but on the flipside, we can notice that the smoothest trend is also shifted towards the right, since the rolling windows (the number of days on which each point is averaged on) is bigger and adapts more slowly to how the temperature changes. In fact, by setting a large $\beta$, we are giving a greater weight to the temperatures that have come before ($v_t$) and a smaller weight to the current temperature.

The equation in $\eqref{eq:ewa}$ is how you implement an **exponentially weighted moving average** or exponentially weighted average for short. The reason why they are called exponentially weighted averages become clear if we look at how $v_n$ is computed, let's take $v_{100}$:

$$
\begin{split}
& v_{100} = 0.1 \theta_{100} + 0.9 & v_{99} & \\
&& \shortparallel \\ &
&  0.1 \theta_{99} + 0.9 & v_{98} &\\
&&& \shortparallel \\
&&&  0.1 \theta_{98} + 0.9 & v_{97} \\
\end{split}
$$

And by expanding the algebra we can see that

$$
v_{100} = 0.1 \cdot 0.9 \theta_{99} + 0.1 \cdot (0.9)^2 \theta_{98} + 0.1 \cdot (0.9)^3 \theta_{97} + 0.1 \cdot (0.9)^4 \theta_{96} \dots
$$

So this is a weighted sum of $\theta$s where the weight of $\theta$ increases exponentially with the steps back from the current $\theta$. We can visualize that in the <a href="fig:ew">figure below</a>


    

<figure id="fig:ew">
    <img src="{{site.baseurl}}/pages/ML-30-DeepLearningSofisticatedOpt_files/ML-30-DeepLearningSofisticatedOpt_7_0.png" alt="png">
    <figcaption>Figure 57. Exponential weights (B) applied to each data point (A) in the exponentially weighted average algorithm</figcaption>
</figure>

In order to say that $\beta = 0.9$ corresponds to averaging over around 10 days we observe that $0.9^{10} \approx 0.35 \approx \frac{1}{e}$, and more in general we have $(1 - \epsilon)^{\frac{1}{\epsilon}} = \frac{1}{e}$ (where $\epsilon = 1-\beta$).

All the terms $(1-\beta) \cdot \beta^x \approx 1$ up to a detail called bias correction. Especially around the first values of $v$, the approximation will be greatly underestimating the data since it cannot base on the full size of the window.

In order to reduce that error we use a corrected version of $v_t$

$$
\frac{v_t}{1-\beta^t}
$$

## Gradient descent with momentum
Gradient descent with momentum (aka just momentum) works almost always faster than regular gradient descent. The basic idea of momentum is to compute the exponentially weighted average of your gradients to update the weights.

Consider <a href="#fig:momentum">Figure 58</a>: we can intuitively understand that larger steps in the $y$ direction will slow down learning, while larger steps in the $x$ direction will speed up learning.  


    

<figure id="fig:momentum">
    <img src="{{site.baseurl}}/pages/ML-30-DeepLearningSofisticatedOpt_files/ML-30-DeepLearningSofisticatedOpt_10_0.png" alt="png">
    <figcaption>Figure 58. Gradient descent steps across the feature space for normal gradient descent (blue) and gradient descent with momentum (B)</figcaption>
</figure>

When implementing momentum, on each iteration $t$, you compute the gradients $dw, db$ on the batch (or mini-batch) and then the exponentially weighted average of the gradients:

$$
\begin{split}
 v_{dw} & = \beta v_{dw} + (1-\beta)dw \\
 v_{db} & = \beta v_{db} + (1-\beta)db \\
\end{split}
$$

and consequently update the parameters matrix $w$ with $v_{dw}$

$$
\begin{split}
w & := w - \alpha v_{dw}\\
b & := b - \alpha v_{db}\\
\end{split}
$$

The effect of this is to smooth out the direction of your gradient descent. Usually bias correction is not necessary.

##  RMSprop
Root mean square prop is another algorithm that will speed up learning. To illustrate how RMSprop works let's consider the feature space in <a href="#fig:rmsprop">Figure 59</a>, where we set the $x$ axis as $w$ and the $y$ axis as $b$ (but they could also be $w_1, w_2$)


```python
fig, ax = plt.subplots(figsize=(10, 4))
x, y = np.mgrid[-2:2:.01, -2:2:.01]
pos = np.dstack((x, y))
rv = multivariate_normal([0, 0], [[1, .05], [0, .7]])
ax.contour(x, y, -rv.pdf(pos))
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('w')
ax.set_ylabel('b')

x = np.linspace(-2, 0)
y = np.zeros(50)
y[::2] = 1
y = (np.sin(y)-.5)*(np.arange(1, 100)[::-2]/75)
sy = pd.Series(y).ewm(alpha=.1).mean().values
ax.plot(x, y, marker='o', markersize=3, alpha=.5, label='gradient descent')
# ax.plot(x, sy, label='gradient descent w/ momentum', marker='o', markersize=3)
```




    [<matplotlib.lines.Line2D at 0x7f4dd56f7880>]




    

<figure id="fig:rmsprop">
    <img src="{{site.baseurl}}/pages/ML-30-DeepLearningSofisticatedOpt_files/ML-30-DeepLearningSofisticatedOpt_13_1.png" alt="png">
    <figcaption>Figure 59. RMSprop</figcaption>
</figure>

In RMSProp, for each iteration $t$ we will compute $s_{dw}$, that are the exponentially weighted average **of the square** of the derivatives.

$$
\begin{split}
 s_{dw} & = \beta s_{dw} + (1-\beta)dw^2 \\
 s_{db} & = \beta s_{db} + (1-\beta)db^2 \\
\end{split}
$$

and the parameters update is also slightly different from momentum

$$
\begin{split}
w & := w - \alpha \frac{dw}{\sqrt{s_{dw}}+\epsilon}\\
b & := b - \alpha \frac{db}{\sqrt{s_{db}}+\epsilon}\\
\end{split}
$$

Where $\epsilon=10^{-8}$ is added for numerical stability.

In order for RMSprop to work we hope that $db$ is large and $dw$ is small, and in fact if we look at <a href="fig:rmsprop">the figure above</a> we can see that the derivatives of the standard gradient descent are much larger in the vertical direction than in the horizontal direction.
