### The problem of overfitting
Let's take some data distributed as in the Figure below


![png](ML-11-Overfitting_files/ML-11-Overfitting_1_0.png)


Three models are shown in the figure Below in panels A, B, C:

* Panel A: We could use linear regression to model this data but this isn't a good model. Looking at the data it seems clear that as $x$ increases, $y$ hits a plateau, while the model implies that $y$ will linearly grow with $x$. We call this problem **underfitting** or we say that the algorithm has **high bias**. It means that the algorithm has a very strong pre-conception (bias) that $y$ are going to behave very linearly.

* Panel B: We could fit a quadratic function and this works pretty well, it seems that the model is "just right".

* Panel C: At the other extreme we fit a 4th order polynomial to the data, and the curve obtained passes exactly through all data points of the training set but it is sort of wiggly and it doesn't seems to be modelling well $y$ behaviour respect $x$. This problem is called **overfitting** or we say that the algorithm has **high variance**. It means that the alogirhtm deosn't generalize well for all possible $x$ values but it seems to cover perfectly only the training set.


![png](ML-11-Overfitting_files/ML-11-Overfitting_3_0.png)


The problem of overfitting comes when we have too many features and the learned hypothesis may fit the training set very well ($J(\theta)\approx0$), but fail to generalize to new examples.

The example above depicted under and overfitting for linear regression but logistic regression can suffer from the same problems.

Until now, in order to choose the degree of the polynomial of our hypothsis we would look at the plotted data. However this is not always possible because the dataset could be too big (million of rows) or too complex (many columns) to visualize. How do we avoid overfitting in those cases?

* Reduce the number of features (at the cost of loosing potentially important information)
    * manually select which features to keep
    * model selection algorithm
* Regularization: Keep all the features, but reduce the magnitude of parameters $\theta_j$. This method works well when we have many features, each of which contributes to predicting $y$
    

# Regularization



```python

```


```python

```
