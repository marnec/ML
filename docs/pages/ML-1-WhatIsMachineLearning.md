---
layout: default
categories: linearRegression
permalink: /ML1/
order: 1
---

# What is Machine Learning

Older, informal definition by Arthur Samuel:

> The field of study that gives computers the ability to learn without being explicitly programmed

More modern definition by Tom Mitchell: 

> A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$

Example: playing checkers.

* $E=$ the experience of playing many games of checkers;
* $T=$ the task of playing checkers;
* $P=$ the probability that the program will win the next game.

## Supervised and Un-supervised learning
A machine learning problem can belong to one of two broad categories:

* **Supervised learning**: We know how the correct output should look like
* **Unsupervised learning**: We don't know how the correct output should look like

## Supervised learning
We know how the correct output should look like. We have the intuition that there is a relationship between the input and the output and ML should identify this relationship.

Supervised learning problems belong in turn to two categories:

* **Regression problems**: Map input to continuous values
* **Classification problems**: Map input to discrete values

### Regression problems
We try to map input variables (features) to some continuous function. We could encode a problem as a regression problem even if output is not striclty continuous ($y \in \mathbb{R}$), provided that there are many possible output values.

Example:

> Given data about the size of houses on the real estate market, try to predict their price. 

Price as a function of size is a psuedo-continuous output (prices in USD have sense only rounded to the second decimal figure), so this is a regression problem.


![png](ML-1-WhatIsMachineLearning_files/ML-1-WhatIsMachineLearning_5_0.png)


### Classification problems
We try to map input variables into discrete categories. 

Example:
> Given a patient with a tumor, we have to predict whether the tumor is malignant or benign. 


![png](ML-1-WhatIsMachineLearning_files/ML-1-WhatIsMachineLearning_7_0.png)


## Unsupervised learning
We don't know how the correct output should look like. Unsupervised learning allows us to approach problems with little or no idea of what our results should look like. Wit unsupervised learning we try to derive structure and patterns from data where we don't necessarily know the effect of the variables. We derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

We can divide unsuperised learning in two categories:

* **Clustering**
* **Non-clustering**

### Clustering
Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.


![png](ML-1-WhatIsMachineLearning_files/ML-1-WhatIsMachineLearning_9_0.png)


#### Difference between clustering and classification


![png](ML-1-WhatIsMachineLearning_files/ML-1-WhatIsMachineLearning_11_0.png)


### Non-clustering
The "Cocktail Party Algorithm" allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).
