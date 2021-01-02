# Model Representation




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
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2104</td>
      <td>399900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1600</td>
      <td>329900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400</td>
      <td>369000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1416</td>
      <td>232000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000</td>
      <td>539900</td>
    </tr>
  </tbody>
</table>
</div>



---
$m = $ number of training examples 


```python
m = len(training_set)
m
```




    47



---
$x =$ input variable / features


```python
x = training_set.sqf
x.head()
```




    0    2104
    1    1600
    2    2400
    3    1416
    4    3000
    Name: sqf, dtype: int64



---
$y =$ output variable / target


```python
y = training_set.price
y.head()
```




    0    399900
    1    329900
    2    369000
    3    232000
    4    539900
    Name: price, dtype: int64



To establish notation for future use, we’ll use $x^{(i)}$ to denote the “input” variables, also called input features, and $y^{(i)}$ to denote the “output” or target variable that we are trying to predict. 

So $x^{(2)}$ is the second (if we count from 1 as it is common in math) or the third (if we count from 0 as it is common in computer science) row of $x$


```python
x[2]  # python starts from 0 so x[2] represents the 3rd row
```




    2400



  
A pair $\left(x^{(i)} , y^{(i)}\right)$ is called a training example, and the dataset that we’ll be using to learn (a list of $m$ training examples) is called a training set. Note that the superscript $^{(i)}$ in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use $X$ to denote the space of input values, and $Y$ to denote the space of output values. In this example, $X = Y = \mathbb{R}$. 

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $h : X \to Y$ so that $h(x)$ is a “good” predictor for the corresponding value of $y$. For historical reasons, this function $h$ is called a hypothesis. Seen pictorially, the process is therefore like this:

![ML](./data/img/ML-flowchart.png)

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When $y$ can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

### How do we represent $h$?
The output of a machine learning algorythm is a function $h$. But how do we represent $h$?. Our initial choice is to represent $h$ as a linear function:

$$y = h_\theta(x) = \theta_0 + \theta_1x$$

Sometimes, we might want to use more complex non-linear functions, but this is the simplest building block of regression algorithms which can be built upon later.


![png](ML-2-ModelAndCostFunction_files/ML-2-ModelAndCostFunction_12_0.png)


where $\theta_0$ is the offset of $y$ from 0; $\theta_1$ is the slope of the line, since it scales how much $y$ varies compared to $x$.

# Cost Function
We can measure the accuracy of our hypothesis function by using a cost function. The cost function calculates the **distance** of predicted data from observed data, obviously we want our predicted data to be as close as possible to truth values, in other words we want to **minimize** the distance from truth values.

This version of cost function takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from $x$ and the actual output $y$.

$$\begin{align}
J(\theta_0,\theta_1) & = \frac{1}{2m}\sum^m_{i=1}\left(\hat{y}_i-y_i\right)^2 \\
& = \frac{1}{2m}\sum^m_{i=1}\left(h_{\theta}(x_i) - y_i \right)^2 \\
& = \frac{1}{2m}\sum^m_{i=1}\left(\left(\theta_0 + \theta_{1}x^{(i)}\right) - y_i \right)^2 
\end{align}$$

To break it apart, it is $\frac{1}{2} \bar{x}$ where $\bar{x}$ is the mean of the squares of $h_\theta (x_{i}) - y_{i}$ or the difference (**distance**) between the predicted value and the actual value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $\left(\frac{1}{2}\right)$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term.

In Python it's simply implemented as:


```python
J = lambda p, y : np.average(np.square((p - y)), axis=0) / 2
```

---
The idea is to chose $\theta_0, \theta_1$ to so that $h_\theta(x)$ is close to $y$ for each training example $(x,y)$. In other words we want to chose $\theta_0, \theta_1$ to minimize the cost function $J\left(\theta_0, \theta_1 \right)$

---
If we try to think of it in visual terms, our training data set is scattered on the $x,y$ plane. We are trying to make a straight line (defined by $h_\theta(x)$) which passes through these scattered data points. 

Our objective is to get the best possible line. The best possible line will that line for which the average squared vertical distance of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of $J(\theta_0, \theta_1)$ would be 0. The following example shows the ideal situation where we have a cost function of 0. 

For a simplified version of the regression hypothesis $h_\theta(x)$ where we removed the offset ($\theta_0$):

$$h_\theta(x)=\theta_1x$$


![png](ML-2-ModelAndCostFunction_files/ML-2-ModelAndCostFunction_21_0.png)


$$
\begin{align}
J(\theta_1) &= \frac{1}{2m}(0^2+0^2+0^2) \\
&= 0
\end{align}
$$

When $\theta_1 = 1$, we get a slope of 1 which goes through every single data point in our model. Conversely, when $\theta_1 = 0.5$, we see the vertical distance from our fit to the data points increase. 


![png](ML-2-ModelAndCostFunction_files/ML-2-ModelAndCostFunction_24_0.png)


Plotting 15 $\theta_1$ values in the interval $[-0.5, 2.5]$ yields a bell shaped graph 


![png](ML-2-ModelAndCostFunction_files/ML-2-ModelAndCostFunction_26_0.png)


Thus as a goal, we should try to minimize the cost function. In this case, $\theta_1 = 1$ is our global minimum. 


```python
# fig, ax = plt.subplots()
O1 = np.linspace(-50, 50, 100)
O0 = np.linspace(-1000, 2000, 100)
x = training_set.sqf
y = training_set.price
X = np.full((len(O1), len(x)), x)
Y = np.full((len(O1), len(y)), y)
P = (X.T * O1).T

from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
# X_scaled = scaler.fit_transform()
X_scaled = np.column_stack([np.ones_like(x), x])

# j = np.zeros(shape=(O0.size, O1.size))
# for i, o0 in enumerate(O0):
#     for u, o1 in enumerate(O1):
#         j[i, u] = J(np.dot(X_scaled, [o0, o1]), y)
# #         j[i, u] = J(np.dot, y)

# # plt.plot(O1, J(P.T, Y.T), marker='o')
# plt.contour(O0, O1, j)
```




    array([[ 0.00000000e+00,  1.31415422e-01],
           [ 0.00000000e+00, -5.09640698e-01],
           [ 0.00000000e+00,  5.07908699e-01],
           [ 0.00000000e+00, -7.43677059e-01],
           [ 0.00000000e+00,  1.27107075e+00],
           [ 0.00000000e+00, -1.99450507e-02],
           [ 0.00000000e+00, -5.93588523e-01],
           [ 0.00000000e+00, -7.29685755e-01],
           [ 0.00000000e+00, -7.89466782e-01],
           [ 0.00000000e+00, -6.44465993e-01],
           [ 0.00000000e+00, -7.71822042e-02],
           [ 0.00000000e+00, -8.65999486e-04],
           [ 0.00000000e+00, -1.40779041e-01],
           [ 0.00000000e+00,  3.15099326e+00],
           [ 0.00000000e+00, -9.31923697e-01],
           [ 0.00000000e+00,  3.80715024e-01],
           [ 0.00000000e+00, -8.65782986e-01],
           [ 0.00000000e+00, -9.72625673e-01],
           [ 0.00000000e+00,  7.73743478e-01],
           [ 0.00000000e+00,  1.31050078e+00],
           [ 0.00000000e+00, -2.97227261e-01],
           [ 0.00000000e+00, -1.43322915e-01],
           [ 0.00000000e+00, -5.04552951e-01],
           [ 0.00000000e+00, -4.91995958e-02],
           [ 0.00000000e+00,  2.40309445e+00],
           [ 0.00000000e+00, -1.14560907e+00],
           [ 0.00000000e+00, -6.90255715e-01],
           [ 0.00000000e+00,  6.68172729e-01],
           [ 0.00000000e+00,  2.53521350e-01],
           [ 0.00000000e+00,  8.09357707e-01],
           [ 0.00000000e+00, -2.05647815e-01],
           [ 0.00000000e+00, -1.27280274e+00],
           [ 0.00000000e+00,  5.00114703e-02],
           [ 0.00000000e+00,  1.44532608e+00],
           [ 0.00000000e+00, -2.41262044e-01],
           [ 0.00000000e+00, -7.16966387e-01],
           [ 0.00000000e+00, -9.68809863e-01],
           [ 0.00000000e+00,  1.67029651e-01],
           [ 0.00000000e+00,  2.81647389e+00],
           [ 0.00000000e+00,  2.05187753e-01],
           [ 0.00000000e+00, -4.28236746e-01],
           [ 0.00000000e+00,  3.01854946e-01],
           [ 0.00000000e+00,  7.20322135e-01],
           [ 0.00000000e+00, -1.01841540e+00],
           [ 0.00000000e+00, -1.46104938e+00],
           [ 0.00000000e+00, -1.89112638e-01],
           [ 0.00000000e+00, -1.01459959e+00]])




```python

```


```python

```
