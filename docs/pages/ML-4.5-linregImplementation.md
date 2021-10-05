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


{% include codeHeader.html %}
```python
import pandas as pd
```

We read data from a `csv` file and cast it into a `pandas.DataFrame`


{% include codeHeader.html %}
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


{% include codeHeader.html %}
```python
xy = df.values.T
X = xy[:-1].T
y = xy[-1]
```

Where the features $X$ are


{% include codeHeader.html %}
```python
X[:5]
```




    array([[2104,    3],
           [1600,    3],
           [2400,    3],
           [1416,    2],
           [3000,    4]])



and their labels $y$


{% include codeHeader.html %}
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


{% include codeHeader.html %}
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


{% include codeHeader.html %}
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


{% include codeHeader.html %}
```python
from sklearn.linear_model import LinearRegression
```

Now we can build the model buy instantiating the `LinearRegression` class


{% include codeHeader.html %}
```python
linreg = LinearRegression()
```

the `linreg` variable contains a linear regression object that allow the computation of the model, but we didn't feed the data to it. Data is fed to the `.fit()` method


{% include codeHeader.html %}
```python
linreg = linreg.fit(X_simple, y)
```

The parameters and bias of the model are returned with


{% include codeHeader.html %}
```python
linreg.coef_, linreg.intercept_
```




    (array([134.52528772]), 71270.49244872917)




    
![svg](ML-4.5-linregImplementation_files/ML-4.5-linregImplementation_23_0.svg)
    


### Multiple Features
We can now introduce the dataset split step that we oversaw in the previous example. In `scikit-learn` splitting the dataset in train and test set is taken care of for us through a function. The proportion of the split can be configured through its arguments.


{% include codeHeader.html %}
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

We can now fit the all ($n=2$) features of the training set $X^t$


{% include codeHeader.html %}
```python
linreg = LinearRegression().fit(X_train, y_train)
```

Since this time $X^t \in \mathbb{R}^{m \times 2}$, we have 2 weight parameters and 1 bias parameter


{% include codeHeader.html %}
```python
linreg.coef_, linreg.intercept_
```




    (array([ 112.70999509, 1927.44419957]), 118980.9276634129)



Parameters fitted on the training set can be used to produce prediction from the test set features


{% include codeHeader.html %}
```python
y_pred = linreg.predict(X_test)
y_pred
```




    array([356619.09445432, 264072.81419883, 478334.51487332, 420751.08166304,
           347827.71483696, 233618.3669574 , 345348.09494488, 633333.50669408,
           260353.38436072, 293151.99293319, 337785.15099058, 323921.82159397,
           280303.05349243, 366988.41400301, 285600.42326187, 335429.61537657])



Predictions can be now compared to the labels of the test set


{% include codeHeader.html %}
```python
from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)
```




    0.8286812907113245



## Pytorch
Whereas `scikit-learn` is a high-level library, `Pytorch` is has a much lower-level approach. Many of the things that in `scikit-learn` happen under the hoods, in `Pytorch` need to be done manually.

The steps for training a model in Pytorch as defined in [Pytorch documentation](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py) are

1. Load Dataset
2. Make Dataset Iterable
3. Create Model Class
4. Instantiate Model Class
5. Instantiate Loss Class
6. Instantiate Optimizer Class
7. Train Model

The main entry point of the framework is the `torch` module


{% include codeHeader.html %}
```python
import torch
```

The first noticeable Pytorch feature is that it works using a proprietary data-structure, called a `tensor`. The underlying mathematical concept of tensor is beyond the scope of this article but can be consulted at the [Wikipedia tensor entry](https://en.wikipedia.org/wiki/Tensor). In Pytorch, a `tensor` is (citing the [pytorch tensor documentation](https://pytorch.org/docs/stable/tensors.html)) *a multi-dimensional matrix containing elements of a single data type*.


{% include codeHeader.html %}
```python
X_tensor = torch.tensor(X, dtype=torch.float32)
X_tensor[:5]
```




    tensor([[2.1040e+03, 3.0000e+00],
            [1.6000e+03, 3.0000e+00],
            [2.4000e+03, 3.0000e+00],
            [1.4160e+03, 2.0000e+00],
            [3.0000e+03, 4.0000e+00]])



As you can notice we had to specify `dtype=np.float32`. This is because the underlying implementation of forward and backward propagation used by Pytorch under the hood would not work with the `int` type.

Furthermore, $y$ tensor would be 1D but this would not comply with requirements of Pytorch methods used below, so we transform it into a column vector with the `unsqueeze(-1)` method. This is equivalent to calling `.reshape(-1, 1)` on a `numpy.array`


{% include codeHeader.html %}
```python
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
y_tensor[:5]
```




    tensor([[399900.],
            [329900.],
            [369000.],
            [232000.],
            [539900.]])



Since data is in very different scales we need to first normalize it. Here we use [standardization](https://en.wikipedia.org/wiki/Standard_score), which rescales data to have mean $\mu=0$ and standard deviation $\sigma=1$

$$
X_\text{std} = \frac{X - \mu}{\sigma}
$$


{% include codeHeader.html %}
```python
X_tensor_norm = (X_tensor - X_tensor.mean()) / torch.sqrt(X_tensor.var())
X_tensor_norm[:5]
```




    tensor([[ 0.9590, -0.8692],
            [ 0.5204, -0.8692],
            [ 1.2166, -0.8692],
            [ 0.3603, -0.8701],
            [ 1.7387, -0.8684]])



A linear regression model can be built using the `Linear` class from the `nn` module, which initializes bias and weights automatically. Its constructor takes as input the number of columns of the input ($n_X$) and of the output ($n_y$)


{% include codeHeader.html %}
```python
model = torch.nn.Linear(2, 1)
```

Training the model will require some hyperparameters that we will define in advance for convenience:

* `epochs` is the number of times the model will see all of our training samples;
* `alpha` is the learning rate, which defines how big are the steps takes in updating the parameters;
* `loss_func` is the loss function $\mathcal{L}$ used at training time. In this case we are using the `MSELoss`, which measures the mean squared error (squared L2 norm) between each element in the input $x$ and target $y$;
* `optim` is the optimization algorithm used. In this case we are using `SGD` (Stochastic Gradient Descent). SGD requires us to select the correct $\alpha$. Up to this point we haven't seen that other optimization algorithm exist that automatically adapt $\alpha$ to data (e.g. [ADAM](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)), so we are sticking with standard SGD.


{% include codeHeader.html %}
```python
epochs = 10
alpha = 0.01
loss_func = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=alpha)
```

Now we need to manually run over the epochs and trigger the update of the parameters that will ultimately produce a fitted model


{% include codeHeader.html %}
```python
from torch.autograd import Variable

for epoch in range(epochs):
    inputs = Variable(X_tensor_norm)
    labels = Variable(y_tensor)
    # Clear gradient buffers because from previous epoch
    optim.zero_grad()
    # get output from the model, given the inputs
    outputs = model(inputs)
    # get loss for the predicted output
    loss = loss_func(outputs, labels)
    # get gradients w.r.t to parameters
    loss.backward()
    # update parameters
    optim.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
```

    epoch 0, loss 131183058944.0
    epoch 1, loss 117986451456.0
    epoch 2, loss 106161905664.0
    epoch 3, loss 95566716928.0
    epoch 4, loss 86073114624.0
    epoch 5, loss 77566492672.0
    epoch 6, loss 69944254464.0
    epoch 7, loss 63114424320.0
    epoch 8, loss 56994611200.0
    epoch 9, loss 51510988800.0



{% include codeHeader.html %}
```python
model(inputs)[:5]
```




    tensor([[150347.1562],
            [124711.3125],
            [165403.1250],
            [115392.8125],
            [195881.3750]], grad_fn=<SliceBackward>)


