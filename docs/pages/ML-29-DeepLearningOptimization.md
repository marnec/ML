---
layout: default
title: "Deep Learning - Optimization"
categories: deeplearning
permalink: /ML29/
order: 29
comments: true
---

# Mini-batch gradient descent

When training a deep-learning model, you training set might very large and slow down your training. In order to prevent this problem what is usually done is to split your training set in **mini-batches**. For example you might split a 5000000 examples training set in 1000 mini-batches ($t$) of 5000 training examples each. You would have 1000 feature vectors $X^{\{t\}}, Y^{\{t\}}$.

From here you would proceed by iterating over your 1000 mini-batches in each training epoch. Below you can see some pseudocode representing the process, where I focus on epoch and mini-batches and remain less rigorous on layers:

```python
for epoch in range(n_iterations):
    for t in range(n_minibatches):
        a = forward_prop(w, x[t], b)
        J[t] = compute_cost(a, y[t])
        dw = backprop(x[t], y[t])
        w := update_weigths(dw)
```

Whereas in batches you expect the value of the cost function $J$ to monotonically decrease with the number of iterations and if this doesn't happen is a signal of some error in the implementation of gradient descent, in mini-batch gradient descent for each $t$ we could have a local increase or decrease f $J$, depending on how hard the examples in the mini-batch are. There should still be a trend down at the increase of $t$.


    

<figure id="minibatchcost">
    <img src="{{site.baseurl}}/pages/ML-29-DeepLearningOptimization_files/ML-29-DeepLearningOptimization_2_0.png" alt="png">
    <figcaption>Figure 55. The value of J over many epochs and over many mini-batches</figcaption>
</figure>


```python

```
