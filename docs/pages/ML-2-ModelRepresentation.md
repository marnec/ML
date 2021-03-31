---
layout: default
title: "Linear Regression - Model Representation"
categories: linearRegression
permalink: /ML2/
order: 2
comments: true
---

# Model Representation
Let's use as an example the housing prices in Portland Oregon and plot them with respect to the surface of the house in square feets (<a href="#fig:housepricescatter">figure below</a>)


    
![png](ML-2-ModelRepresentation_files/ML-2-ModelRepresentation_2_0.png)
    


<i id="fig:housepricescatter">Scatterplot of house prices as a function of the living area in square foots</i>

Let's say that you want to know the possible price of an house given its surface. One thing that you could do is to draw a straight line that describes the growth of prices with surface (<a href="#fig:linreghouseprices"></a>).


    
![png](ML-2-ModelRepresentation_files/ML-2-ModelRepresentation_4_0.png)
    


<i id="fig:linreghouseprices">A possible description of the dependence of house prices from square foots</i>

This is an example of supervised learning since we know the answer for each example in the dataset and it is also an example of regression problem, where we try to predict a contnuous value.

In supervised learning we have a training set that contains examples $(x, y)$ and our job is learning how to predict labels $y$ for new examples $x$




<style  type="text/css" >
#T_3ddd7_row0_col0,#T_3ddd7_row0_col1,#T_3ddd7_row1_col0,#T_3ddd7_row1_col1,#T_3ddd7_row2_col0,#T_3ddd7_row2_col1,#T_3ddd7_row3_col0,#T_3ddd7_row3_col1,#T_3ddd7_row4_col0,#T_3ddd7_row4_col1{
            text-align:  left;
        }</style><table id="T_3ddd7_" ><thead>    <tr>        <th class="col_heading level0 col0" >sqf</th>        <th class="col_heading level0 col1" >price</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_3ddd7_row0_col0" class="data row0 col0" >2104</td>
                        <td id="T_3ddd7_row0_col1" class="data row0 col1" >399.900000</td>
            </tr>
            <tr>
                                <td id="T_3ddd7_row1_col0" class="data row1 col0" >1600</td>
                        <td id="T_3ddd7_row1_col1" class="data row1 col1" >329.900000</td>
            </tr>
            <tr>
                                <td id="T_3ddd7_row2_col0" class="data row2 col0" >2400</td>
                        <td id="T_3ddd7_row2_col1" class="data row2 col1" >369.000000</td>
            </tr>
            <tr>
                                <td id="T_3ddd7_row3_col0" class="data row3 col0" >1416</td>
                        <td id="T_3ddd7_row3_col1" class="data row3 col1" >232.000000</td>
            </tr>
            <tr>
                                <td id="T_3ddd7_row4_col0" class="data row4 col0" >3000</td>
                        <td id="T_3ddd7_row4_col1" class="data row4 col1" >539.900000</td>
            </tr>
    </tbody></table>



* $m$ denotes the number of training examples 
* $x$ denotes the input variables / features
* $y$ denotes the output variable / target
* $x, y$ is a training example
* $^{(i)}$ is the index in the training set (# row) 
* $x^{(i)}, y^{(i)}$ is the specific training example at row $i$

So $x^{(2)}$ is the second (if we count from 1 as it is common in math) or the third (if we count from 0 as it is common in computer science) row of $x$

  
A pair $\left(x^{(i)} , y^{(i)}\right)$ is called a training example, and the dataset that we’ll be using to learn (a list of $m$ training examples) is called a training set. Note that the superscript $^{(i)}$ in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use $X$ to denote the space of input values, and $Y$ to denote the space of output values. In this example, $X = Y = \mathbb{R}$. 

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $h : X \to Y$ so that $h(x)$ is a “good” predictor for the corresponding value of $y$. For historical reasons, this function $h$ is called a hypothesis. Seen pictorially, the process is therefore like this:

![ML](./data/img/ML-flowchart.png)

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When $y$ can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

## How to repesent a linear regression hypothesis
The output of a machine learning algorythm is a function $h$. But how do we represent $h$?. Our initial choice is to represent $h$ as a linear function:

$$y = h_\theta(x) = \theta_0 + \theta_1x$$

Sometimes, we might want to use more complex non-linear functions, but this is the simplest building block of regression algorithms which can be built upon later.


    
![png](ML-2-ModelRepresentation_files/ML-2-ModelRepresentation_11_0.png)
    


where $\theta_0$ is the offset of $y$ from 0; $\theta_1$ is the slope of the line, since it scales how much $y$ varies compared to $x$.

# Non-linear regression
We can improve our features and the form of our hypothesis function in a couple different ways. We can combine multiple features into one. For example, we can combine $x_1$ and $x_2$ into a new feature $x_3$ by taking $x_1 \dot x_2x$

## Polynomial Regression
Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is $h_\theta(x) = \theta_0 + \theta_1 x_1$ then we can create additional features based on $x_1$, to get the quadratic function $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2$ or the cubic function $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3$. In the cubic version, we have created new features $x_2$ and $x_3$ where $x_2 = x_1^2$ and $x_3 = x_1^3$. 

To make it a square root function, we could do: $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}$

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

eg. if $x_1$ has range $[1, 1\,000]$ then range of $x_1^2$ becomes $[1, 1\,000\,000]$ and that of $x_1^3$ becomes $[1, 1\,000\,000\,000]$
