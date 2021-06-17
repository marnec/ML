---
layout: default
categories: linearRegression
title: "Linear Regression - Implementation"
permalink: /ML4.5/
order: 4.5
comments: true
---

# Linear regression implementation
Since linear regression is a trivial model, it is relatively easy to implement it from scratches and maybe in the future I'll implement a full version on this page. 

Many libraries enabling a user to build and train a linear regression model exist. In the last years I feel like `scikit-learn` and `pytorch` are the most widely used libraries in machine learning.

## Reading data
For this example we are using house prices as a function of inhabitable surface and number of rooms. Data is stored in a csv file, to parse it into a python data structure we use `pandas`. This is a preliminary step for any approach and while some libraries may offer custom way to parse data I find that this is just better. Delegating parsing to a second library follows the *single-responsibility* principle. This is at least true for datasets saved in common formats like `csv` or `tsv` or similar. Sometimes we will deal with custom formats like Pytorch's `pt` files: in that case it is obviously better (or sometimes necessary) to take care of data loading with the right library.


```python
import pandas as pd
```

We read data from a `csv` file and cast it into a `pandas.DataFrame`


```python
df = pd.read_csv('./data/house_pricing.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sqf</th>
      <th>rooms</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2104</td>
      <td>3</td>
      <td>399900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1600</td>
      <td>3</td>
      <td>329900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400</td>
      <td>3</td>
      <td>369000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1416</td>
      <td>2</td>
      <td>232000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000</td>
      <td>4</td>
      <td>539900</td>
    </tr>
  </tbody>
</table>
</div>



This dataset has two feature columns (`sqf` and `rooms`) and a label column (`price`)

Let's assign the features $X$ and the labels $y$ to two different variables


```python
xy = df.values.T
X = xy[:-1].T
y = xy[-1]
```

Where the features $X$ are


```python
X[:5]
```




    array([[2104,    3],
           [1600,    3],
           [2400,    3],
           [1416,    2],
           [3000,    4]])



and their labels $y$


```python
y[:5].reshape(-1, 1)
```




    array([[399900],
           [329900],
           [369000],
           [232000],
           [539900]])



## scikit-learn
Linear regression in `scikit-learn` is as easy as one line of code. To keep this first example as easy as possible, I'm not going to split the data in training and dev sets. I'm just fitting the model to the whole dataset. In a real scenario, there should be a preliminary step of dataset splitting. 

### Single feature
In order for the first example to be as simple as possible and plottable, for now we drop the `rooms` column from the features and we are only left with the `sqf` column. This means that in this first example we are exploring linear dependency between the inhabitable surface and the price of a house.


```python
X_simple = X[:, 0]
X_simple
```




    array([2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494, 1940,
           2000, 1890, 4478, 1268, 2300, 1320, 1236, 2609, 3031, 1767, 1888,
           1604, 1962, 3890, 1100, 1458, 2526, 2200, 2637, 1839, 1000, 2040,
           3137, 1811, 1437, 1239, 2132, 4215, 2162, 1664, 2238, 2567, 1200,
            852, 1852, 1203])



Since the `fit()` function that we are using later wants a 2D-vector of shape $(m, n)$  and we only have one feature, we need to reshape the array in the form $(m, 1)$. On the other hand $y$ can either be a 2D or 1D array.


```python
X_simple = X_simple.reshape(-1, 1)
X_simple[:5]
```




    array([[2104],
           [1600],
           [2400],
           [1416],
           [3000]])



Building a linear regression model with [scikit-learn](https://scikit-learn.org/) requires the `LinearRegression` class


```python
from sklearn.linear_model import LinearRegression
```

Now we can build the model buy instantiating the `LinearRegression` class


```python
linreg = LinearRegression()
```

the `linreg` variable contains a linear regression object that allow the computation of the model, but we didn't feed the data to it. Data is fed to the `.fit()` method


```python
linreg = linreg.fit(X_simple, y)
```

The parameters and bias of the model are returned with


```python
linreg.coef_, linreg.intercept_
```




    (array([134.52528772]), 71270.49244872917)




    
![png](ML-4.5-linregImplementation_files/ML-4.5-linregImplementation_23_0.png)
    


### Multiple Features
We can now introduce the dataset split step that we oversaw in the previous example. In `scikit-learn` splitting the dataset in train and test set is taken care of for us through a function. The proportion of the split can be configured through its arguments.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

We can now fit the all ($n=2$) features of the training set $X^t$


```python
linreg = LinearRegression().fit(X_train, y_train)
```

Since this time $X^t \in \mathbb{R}^{m \times 2}$, we have 2 weight parameters and 1 bias parameter


```python
linreg.coef_, linreg.intercept_
```




    (array([ 121.23967719, -533.38308167]), 102899.49800253182)



Parameters fitted on the training set can be used to produce prediction from the test set features


```python
y_pred = linreg.predict(X_test)
y_pred
```




    array([223605.79211291, 320331.02107066, 315529.85835588, 481628.21610898,
           572921.69303459, 246786.96138798, 362886.14776507, 205128.93680682,
           464484.997252  , 411988.21702784, 392274.57401844, 643143.85706018,
           268610.10328255, 282431.42648245, 368026.63858003, 338638.21232666])



Predictions can be now compared to the labels of the test set


```python
from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)
```




    0.8322693778910275



## Pytorch
Whereas `scikit-learn` is a high-level library, `Pytorch` is has a much lower-level approach. Many of the things that in `scikit-learn` happen under the hoods, in `Pytorch` need to be done manually. 

The main entry point of the framework is the `torch` module


```python
import torch
```

The first noticeable `Pytorch` feature is that it works using a proprietary data-structure, called a `tensor`. The underlying mathematical concept of tensor is beyond the scope of this article but can be consulted at the [Wikipedia tensor entry](https://en.wikipedia.org/wiki/Tensor). In Pytorch, a `tensor` is (citing the [pytorch tensor documentation](https://pytorch.org/docs/stable/tensors.html)) *a multi-dimensional matrix containing elements of a single data type*.


```python
torch.tensor(X, dtype=torch.float32)[:5]
```




    tensor([[2.1040e+03, 3.0000e+00],
            [1.6000e+03, 3.0000e+00],
            [2.4000e+03, 3.0000e+00],
            [1.4160e+03, 2.0000e+00],
            [3.0000e+03, 4.0000e+00]])




```python
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        return self.linear(x)
```


```python
model = linearRegression(2, 1)
```


```python
epochs = 100
alpha = 0.01
loss_func = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=alpha)
```


```python
for epoch in range(epochs):
    inputs = Variable(torch.from_numpy(X.astype(np.float32)))
    labels = Variable(torch.from_numpy(y.reshape(-1, 1).astype(np.float32)))
    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optim.zero_grad()
    # get output from the model, given the inputs
    outputs = model(inputs)
    # get loss for the predicted output
    loss = loss_func(outputs, labels)
    # get gradients w.r.t to parameters
    loss.backward()
    # update parameters
    optim.step()
#     print('epoch {}, loss {}'.format(epoch, loss.item()))
```
