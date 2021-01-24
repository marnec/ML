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


When we have many features it becomes impossible to plot hypotheses. How do we tell if our hypothesis is overfitting? The standard way to evaluate a training hypothesis is to split the training set in two randomly selected subsets. The first subset ($70\%$ of the examples) will be the **training set** and the second subset ($30\%$ of the examples) will be the **test set** (blue background).




<style  type="text/css" >
#T_86088242_5e5c_11eb_a487_40a3cc65d4e3row7_col0,#T_86088242_5e5c_11eb_a487_40a3cc65d4e3row7_col1,#T_86088242_5e5c_11eb_a487_40a3cc65d4e3row8_col0,#T_86088242_5e5c_11eb_a487_40a3cc65d4e3row8_col1,#T_86088242_5e5c_11eb_a487_40a3cc65d4e3row9_col0,#T_86088242_5e5c_11eb_a487_40a3cc65d4e3row9_col1{
            background-color:  lightskyblue;
        }</style><table id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3" ><thead>    <tr>        <th class="col_heading level0 col0" >sqf</th>        <th class="col_heading level0 col1" >price</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row0_col0" class="data row0 col0" >2104</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row0_col1" class="data row0 col1" >399900</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row1_col0" class="data row1 col0" >1600</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row1_col1" class="data row1 col1" >329900</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row2_col0" class="data row2 col0" >2400</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row2_col1" class="data row2 col1" >369000</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row3_col0" class="data row3 col0" >1416</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row3_col1" class="data row3 col1" >232000</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row4_col0" class="data row4 col0" >3000</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row4_col1" class="data row4 col1" >539900</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row5_col0" class="data row5 col0" >1985</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row5_col1" class="data row5 col1" >299900</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row6_col0" class="data row6 col0" >1534</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row6_col1" class="data row6 col1" >314900</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row7_col0" class="data row7 col0" >1427</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row7_col1" class="data row7 col1" >198999</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row8_col0" class="data row8 col0" >1380</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row8_col1" class="data row8 col1" >212000</td>
            </tr>
            <tr>
                                <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row9_col0" class="data row9 col0" >1494</td>
                        <td id="T_86088242_5e5c_11eb_a487_40a3cc65d4e3row9_col1" class="data row9 col1" >242500</td>
            </tr>
    </tbody></table>



And we will differentiate the notation between test set and training set as follows:

Training set:

$$
\begin{align}
&\left(x^{(1)}, y^{(1)} \right)\\
&\left(x^{(2)}, y^{(2)} \right)\\ 
& \quad \;\; \vdots \\ 
&\left(x^{(m)}, y^{(m)} \right)
\end{align}
\label{eq:trainingdata} \tag{1}
$$

Test set:

$$
\begin{align}
&\left(x^{(1)}_{\text{test}}, y^{(1)}_{\text{test}} \right)\\
&\left(x^{(2)}_{\text{test}}, y^{(2)}_{\text{test}} \right)\\ 
& \quad \;\; \vdots \\ 
&\left(x^{(m_\text{test})}_{\text{test}}, y^{(m_\text{test})}_{\text{test}} \right)
\end{align}
\label{eq:tetsdata} \tag{2}
$$

So here's how you will proceed with these two subsets: 

1. Learn paramters $\theta$ from the training set (minimizing error $J(\theta)$)
2. Compute test set error $J_\text{test}(\theta)$

In logistic regression you can of course compute $J_\text{test}(\theta)$ or alternatively you could compute the **misclassification error** (also called 0/1 misclassification error to convey that you can either classify an example correctly or incorrectly).

$$
\begin{equation}
\text{err}(h_\theta(x), y)=
\begin{cases}\!% alignment adjustment
&1 \quad
  \begin{aligned}[t]% adjust case condition placement here, use [t]op, [b]ottom, or [c]enter (default)
    &\text{if } h_\theta(x) \geq 0.5, \; y=0\\ 
    &\text{or } h_\theta(x) < 0.5, \; y=1\\
  \end{aligned} \\ \\
&0 \quad \text{otherwise}
\end{cases}
\end{equation}
$$

$$
\text{Test Error} = \frac{1}{m_\text{test}} \sum^{m_\text{test}}_{i=1} \text{err} \left( h_\theta\left( x^{(i)}_\text{test} \right), y^{(i)}_\text{test} \right)
$$

## Model selection
This section regards model selection problems. In particular we will touch upon how o choose the polynomial features to include in an hypothesis and how to choose the regularization paramter $\lambda$. 

To do this we will split the data not only in training and test sets but in three subsets. The **training-set**, the **test-set** and the **validation-set**.

Let's say that you are trying to chose what degree polynomial to fit to the data. It is as if  in our algorithm there is an extra parameter $d$ to set that represents what degree of polynomial we want to use.

$$
\begin{align}
d=1 \to \; & h_\theta(x) = \theta_0+\theta_1x & \to \theta^{(1)}\\
d=2 \to \; & h_\theta(x) = \theta_0+\theta_1x+\theta_2x^2 & \to \theta^{(2)}\\
d=3 \to \; & h_\theta(x) = \theta_0+\theta_1x+\dots+\theta_3x^3 & \to \theta^{(3)}\\
&\vdots &\\
d=10 \to \; & h_\theta(x) = \theta_0+\theta_1x+\dots+\theta_{10}x^{10} & \to \theta^{(10)}\\
\end{align}
$$

You want to choose a model, fit the model and get an estimate of how well the fitted model generalize on new examples. In order to chose a model we could be tempted to calculate the test-set error for each model with parameter $d$ and fitted parameters $\theta^{(i)}$ and chose the model with the smallest $J_\text{test}(\theta)$.

$$
\begin{align}
\theta^{(1)} \to \; & J_\text{test}\left(\theta^{(1)}\right)\\
\theta^{(2)} \to \; & J_\text{test}\left(\theta^{(2)}\right)\\
\theta^{(3)} \to \; & J_\text{test}\left(\theta^{(3)}\right)\\
&\vdots \\
\theta^{(10)} \to \; & J_\text{test}\left(\theta^{(10)}\right)\\
\end{align}
$$

However, the chosen $J_\text{test}(\theta^{(i)})$ is very likely to be an optimistic estimate of the generalization error because we fit the parameter $d$ to the test-set so it will tend to perform better on the test than on a general case.

In order to resolve this issue we are going to split our dataset in three subsets: training-set, the cross validation-set (or CV, orange background) and the test set (blue background). Usually the training set will take around $60\%$ of your dataset and the test and cross-validation sets will take $20\%$ each.




<style  type="text/css" >
#T_86088243_5e5c_11eb_a487_40a3cc65d4e3row6_col0,#T_86088243_5e5c_11eb_a487_40a3cc65d4e3row6_col1,#T_86088243_5e5c_11eb_a487_40a3cc65d4e3row7_col0,#T_86088243_5e5c_11eb_a487_40a3cc65d4e3row7_col1{
            background-color:  bisque;
        }#T_86088243_5e5c_11eb_a487_40a3cc65d4e3row8_col0,#T_86088243_5e5c_11eb_a487_40a3cc65d4e3row8_col1,#T_86088243_5e5c_11eb_a487_40a3cc65d4e3row9_col0,#T_86088243_5e5c_11eb_a487_40a3cc65d4e3row9_col1{
            background-color:  lightskyblue;
        }</style><table id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3" ><thead>    <tr>        <th class="col_heading level0 col0" >sqf</th>        <th class="col_heading level0 col1" >price</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row0_col0" class="data row0 col0" >2104</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row0_col1" class="data row0 col1" >399900</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row1_col0" class="data row1 col0" >1600</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row1_col1" class="data row1 col1" >329900</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row2_col0" class="data row2 col0" >2400</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row2_col1" class="data row2 col1" >369000</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row3_col0" class="data row3 col0" >1416</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row3_col1" class="data row3 col1" >232000</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row4_col0" class="data row4 col0" >3000</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row4_col1" class="data row4 col1" >539900</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row5_col0" class="data row5 col0" >1985</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row5_col1" class="data row5 col1" >299900</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row6_col0" class="data row6 col0" >1534</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row6_col1" class="data row6 col1" >314900</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row7_col0" class="data row7 col0" >1427</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row7_col1" class="data row7 col1" >198999</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row8_col0" class="data row8 col0" >1380</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row8_col1" class="data row8 col1" >212000</td>
            </tr>
            <tr>
                                <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row9_col0" class="data row9 col0" >1494</td>
                        <td id="T_86088243_5e5c_11eb_a487_40a3cc65d4e3row9_col1" class="data row9 col1" >242500</td>
            </tr>
    </tbody></table>



And of course now, in addition to $\eqref{eq:trainingdata}$ and $\eqref{eq:testdata}$  we will have:


$$
\begin{align}
&\left(x^{(1)}_{\text{CV}}, y^{(1)}_{\text{CV}} \right)\\
&\left(x^{(2)}_{\text{CV}}, y^{(2)}_{\text{CV}} \right)\\ 
& \quad \;\; \vdots \\ 
&\left(x^{(m_{CV})}_{\text{CV}}, y^{(m_{CV})}_{\text{CV}} \right)
\end{align}
$$

And we can also define the errors for the three subsets.

Training error

$$
\begin{equation}
J_\text{train}(\theta) = \frac{1}{2m}\sum^m_{i=1}\left(h_\theta\left(x^{(i)}\right) - y^{(i)}\right)^2
\end{equation}
\label{eq:trainerr} \tag{3}
$$

Cross Validation error

$$
\begin{equation}
J_\text{CV}(\theta) = \frac{1}{2m_\text{CV}}\sum^{m_\text{CV}}_{i=1}\left(h_\theta\left(x_\text{CV}^{(i)}\right) - y_\text{CV}^{(i)}\right)^2
\end{equation}
\label{eq:crosserr} \tag{4}
$$

Test error

$$
J_\text{test}(\theta) = \frac{1}{2m_\text{test}}\sum^{m_\text{test}}_{i=1}\left(h_\theta\left(x_\text{test}^{(i)}\right) - y_\text{test}^{(i)}\right)^2
$$

And so before testing the generalization power of you algorithm on the test set you can select the model that produces $\min_\theta J(\theta)$ calculated on the Cross Validation set.

While it is ill advised to optimizize $J_\text{test}(\theta)$ and $J_\text{CV}(\theta)$ on the same subset (test set), sometimes you will encounter algorithms trained in such a way. If the test is very big this should not be a problem but othwerise you should also have separate test and cross-validation sets.

## Bias vs Variance
When a ML algorith is underperfoming with respect to expectations is almost always because of an **over-fitting** problem or **under-fitting** problem.


![png](ML-17-Diagnostic_files/ML-17-Diagnostic_12_0.png)


Let's say that training error and cross validation error are defined as in $\eqref{eq:trainerr}$ and $\eqref{eq:crosserr}$.


```python
%%capture --no-display
xcross = np.linspace(-.5, .5)
xtrain = np.linspace(-1, 2)

jtrain = -np.log(xtrain)
jcross = 10*xcross ** 2

fig, ax = plt.subplots()
ax.plot(xcross-.8, jcross+.7, label='$J_{CV}$')
ax.plot(xtrain - 1.5 , jtrain, label='$J_{train}$')
ax.axvline(-.8, ls='--', c='k', label='optimal $d$ value')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('Cost $J(\\theta)$')
ax.set_xlabel('$d$ polynome degree')
ax.legend();
```




    <matplotlib.legend.Legend at 0x7fa4e8cd9a90>




![png](ML-17-Diagnostic_files/ML-17-Diagnostic_14_1.png)



```python

```
