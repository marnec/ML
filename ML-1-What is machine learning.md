```python
%pylab --no-import-all inline
import pandas as pd
import seaborn as sns
```

    Populating the interactive namespace from numpy and matplotlib


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

* Supervised learning
* Unsupervised learning

## Supervised learning
we know how the correct output should look like. We have the intuition that there is a relationship between the input and the output and ML should identify this relationship.

Supervised learning problems belong in turn to two categories:

* Regression problems
* Classification problems

### Regression problems
We try to map input variables (features) to some continuous function. We could encode a problem as a regression problem even if output is not striclty continuous ($y \in \mathbb{R}$), provided that there are many possible output values.

Example:

> Given data about the size of houses on the real estate market, try to predict their price. 

Price as a function of size is a psuedo-continuous output (prices in USD have sense only rounded to the second decimal figure), so this is a regression problem.


```python
sns.regplot(data=sns.load_dataset('car_crashes'), x='speeding', y='total');
```


    
![png](ML-1-What%20is%20machine%20learning_files/ML-1-What%20is%20machine%20learning_5_0.png)
    


### Classification problems
We try to map input variables into discrete categories. 

Example:
> Given a patient with a tumor, we have to predict whether the tumor is malignant or benign. 


```python

```
