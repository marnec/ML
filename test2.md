---
layout: default
title: "Machine learning diagnostic"
categories: evaluation
permalink: /ML17/
order: 17
comments: true
---

## Evaluating an hypothesis

In the case of house prices if we want to see how our hypothesis is performing we could just plot it. Since we have just one feature (the area of each house) we can plot the feature against the price.


![png](ML-17-Diagnostic_files/ML-17-Diagnostic_2_0.png)


<a id="figcaption">This is the caption of the figure</a>

When we have many features it becomes impossible to plot hypotheses. How do we tell if our hypothesis is overfitting? The standard way to evaluate a training hypothesis is to split the training set in two randomly selected subsets. The first subset ($70\%$ of the examples) will be the **training set** and the second subset ($30\%$ of the examples) will be the **test set** (blue background).
