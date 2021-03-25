---
layout: default
title: "Deep Learning - Bias & Variance"
categories: deeplearning
permalink: /ML25/
order: 25
comments: true
---

# Bias and Variance
Bias and Variance in ML (<a href="ML8#biasvariance">check this figure</a>)  are fundamental concepts and expert practitioners usually have a deep understanding of bias/variance related topics. 

In the deep learning era there is less discussion about the bias/variance trade-off because in the deep learning era there is less trade-off. So the concepts of bias and variance are still central but their trade-off is no more so important.

When only two features are present we can just look at the model (<a href="ML8#biasvariance">check this figure</a>) and identify situations of high bias (panel A) or high variance (panel C).

When many features are present we can no longer visualize the model but we can employ some metrics that will help us identify these problems.

## Identify bias/variance from subset error
Suppose you have a classifier that should identify cat pictures. So $y=1$ for a picture of a cat and $y=0$ for any other pictures.

Suppose you fit your model on the training set and then measure the error on both the training set and development set and obtain the error as in <a href="#biasvarerror">the table below</a>.




<style  type="text/css" >
</style><table id="T_5c9b1_" id="biasvarerror"><caption>Four cases of error (as percentage of miscalssifications) calculated on the train- and test-sets after fitting a model</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >case 1</th>        <th class="col_heading level0 col1" >case 2</th>        <th class="col_heading level0 col2" >case 3</th>        <th class="col_heading level0 col3" >case 4</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_5c9b1_level0_row0" class="row_heading level0 row0" >train set</th>
                        <td id="T_5c9b1_row0_col0" class="data row0 col0" >1%</td>
                        <td id="T_5c9b1_row0_col1" class="data row0 col1" >15%</td>
                        <td id="T_5c9b1_row0_col2" class="data row0 col2" >15%</td>
                        <td id="T_5c9b1_row0_col3" class="data row0 col3" >0.5%</td>
            </tr>
            <tr>
                        <th id="T_5c9b1_level0_row1" class="row_heading level0 row1" >dev set</th>
                        <td id="T_5c9b1_row1_col0" class="data row1 col0" >11%</td>
                        <td id="T_5c9b1_row1_col1" class="data row1 col1" >16%</td>
                        <td id="T_5c9b1_row1_col2" class="data row1 col2" >30%</td>
                        <td id="T_5c9b1_row1_col3" class="data row1 col3" >1%</td>
            </tr>
    </tbody></table>



Assuming that a person would have an error $\approx 0%$ and that the train and dev sets are drawn from the same distribution:

* case 1 is a case of high variance
* case 2 is a case of high bias
* case 3 is a case of high bias AND high variance (the worst scenario)
* case 4 is a case of low bias and low variance (the best scenario)

It is important to notice that we detected bias and variance based on the assumption that the **optimal error**, also called **Bayes error** is $\approx 0%$.

Would the Bayes error $\approx 15%$, then we can say that case 2 is a case of low bias and low variance.

## Basic recipe for correct training
This is a basic recipe to apply when training a model:


```python
f = Flow(direction='ns', figsize=(4, 5))
f.node('fit the parameters')
f.node('test the model')
f.node('a', label='High bias?\n(training set performance)')
f.node('Y', bbox=dict(boxstyle='circle'), travel='e', distance=.5)
f.node('c', 'bigger network', travel='ne', distance=.5)
f.edge('c', 'a', tailport='n', arrowprops=dict(connectionstyle="angle3,angleA=30,angleB=-110"))
f.node('d', label='train for longer', startpoint='Y', travel='e', distance=.5)
f.edge('d', 'a', tailport='ne', headport='ne', arrowprops=dict(connectionstyle="angle3,angleA=30,angleB=-110"))
f.node('N', bbox=dict(boxstyle='circle'), startpoint='a')
f.node('b', label='High Variance?\n(dev set performance)')
f.node('Y', bbox=dict(boxstyle='circle'), travel='e', distance=.5)
f.node('e', label='get more data', travel='ne', distance=.5)
f.edge('e', 'a', tailport='ne', headport='ne', arrowprops=dict(connectionstyle="angle3,angleA=32,angleB=-110"))
f.node('f', label='regularization', travel='e', distance=.5, startpoint='Y')
f.edge('f', 'a', tailport='ne', headport='ne', arrowprops=dict(connectionstyle="angle3,angleA=32,angleB=-110"))
f.node('g', label='NN architecture', travel='se', distance=.5, startpoint='Y')
f.edge('g', 'a', tailport='ne', headport='ne', arrowprops=dict(connectionstyle="angle3,angleA=32,angleB=-110"))
f.node('N', bbox=dict(boxstyle='circle'), startpoint='b')
f.node('Done')
```




    <mpl_flow.Node at 0x7f32a93ce9d0>




    
![png](ML-25-DeepLearningBiasVariance_files/ML-25-DeepLearningBiasVariance_6_1.png)
    

