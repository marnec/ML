---
layout: default
title: "Skewed classes"
categories: design
permalink: /ML19/
order: 19
comments: true
---

# Error metrics for skewed classes
Of particular importance is correctly evaluating a learning algorithm wen classes are unbalanced (skewed)

Consider a problem where we have to tell if patients have or don't have cancer based on a set of input features.

We train a logistic regression model $h_\theta(x)$ where $y=1$ means that the patient has cancer, $y=0$ means that he/she doesn't.

We then find that the learning algorithm has $1\%$ error on the test set, in other words it outputs $99\%$ of correct diagnoses.

This may look impressive at a first glance but it doesn't anymore if we add a new piece of information: only $0.50\%$ of patients have cancer.

This means that a non-learning algorithm that always outputs $0$ will have $0.50\%$ error.

When one class in our dataset is much more abundant than the other(s) we have a case of dataset unbalance and we say that the classes are skewed. In this cases using classification error may lead to false deductions and is in general a good practice.

## Precision/Recall
When we are dealing with skewed classes we can use the couple of metrics **precision** and **recall**.

These metrics comes from the count of correct and incorrect classification of a learning algorithm on a test set. For the case of our cancer classifier (and for all binary classifiers) classifications will fall in one of four cases, summarized in the **confusion matrix**.




<style  type="text/css" >
</style><table id="T_e80b819c_66ef_11eb_bb2f_40a3cc65d4e3" ><caption>Scheme of the contingency table for the binary case, called a confusion matrix</caption><thead>    <tr>        <th class="blank" ></th>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" colspan=2>Actual class</th>    </tr>    <tr>        <th class="blank" ></th>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" >1</th>        <th class="col_heading level1 col1" >0</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_e80b819c_66ef_11eb_bb2f_40a3cc65d4e3level0_row0" class="row_heading level0 row0" rowspan=2>Predicted class</th>
                        <th id="T_e80b819c_66ef_11eb_bb2f_40a3cc65d4e3level1_row0" class="row_heading level1 row0" >1</th>
                        <td id="T_e80b819c_66ef_11eb_bb2f_40a3cc65d4e3row0_col0" class="data row0 col0" >True positive</td>
                        <td id="T_e80b819c_66ef_11eb_bb2f_40a3cc65d4e3row0_col1" class="data row0 col1" >False positive</td>
            </tr>
            <tr>
                                <th id="T_e80b819c_66ef_11eb_bb2f_40a3cc65d4e3level1_row1" class="row_heading level1 row1" >0</th>
                        <td id="T_e80b819c_66ef_11eb_bb2f_40a3cc65d4e3row1_col0" class="data row1 col0" >False negative</td>
                        <td id="T_e80b819c_66ef_11eb_bb2f_40a3cc65d4e3row1_col1" class="data row1 col1" >True negative</td>
            </tr>
    </tbody></table>



Starting from this tool we can compute precision and recall:

### Precision
Precision answer the question: how many of the selected cases are relevant? To apply it to the cancer classifier it would be: Of all the patients for which $y=1$, what fraction actually has cancer? It is calculated as:

$$
\frac{\text{#True positives}}{\text{#Predicted positives}} = \frac{\text{#True positives}}{\text{#True positives} + \text{#False positives}}
$$

### Recall
Recall answer the question: how many of the relevant cases are selected? To apply it to the cancer classifier it would be: Of all the patients that actually have cancer, what fraction did we correctly detect?

$$
\frac{\text{#True positives}}{\text{#Actual positives}} = \frac{\text{#True positives}}{\text{#True positives} + \text{#False negatives}}
$$

### Trading off precision for recall
Let's say that we have trained a logistic regression algorithm $0 \geq h_\theta(x) \geq 1$ and we predict:

$$
y=
\begin{cases}
1 \quad \text{if } & h_\theta(x) \geq 0.5 \\
0 \quad \text{if } & h_\theta(x) < 0.5
\end{cases}
$$

Since telling a patient that he/she has cancer may cause a great chock in him/her, we want to give this news only if we are very confident of the prediction. So we may want to increase the threshold:

$$
y=
\begin{cases}
1 \quad \text{if } & h_\theta(x) \geq 0.9 \\
0 \quad \text{if } & h_\theta(x) < 0.9
\end{cases}
$$

This way we will attain higher precision but lower recall. However now we want to avoid missing to many cases of actual cancer, in this case we may want to lower the threshold:

$$
y=
\begin{cases}
1 \quad \text{if } & h_\theta(x) \geq 0.3 \\
0 \quad \text{if } & h_\theta(x) < 0.3
\end{cases}
$$

And we will attain higher recall but lower precision.

In general by lowering the threshold we will trade off precision for recall and if we were to plot recall and precision for a number of possible threshold values we would have something like <a href="#prerec">Figure 3</a>:


    

<figure id="prerec">
    <img src="{{site.baseurl}}/pages/ML-19-SkewedClasses_files/ML-19-SkewedClasses_6_0.png" alt="png">
    <figcaption>Figure 3. An ideal precision-recall curve for decreasing values of threshold (lef-to-right) applied to a logistic regression output scores compared to the actual classes.</figcaption>
</figure>

### A single metric F1-Score
Precision and recall are reliable metrics that complement each other in telling the performance of a learning algorithm. But how do we compare precision and recall? 

In the table below we have the values of precision and recall for three versions of an algorithm, and we would like to have a single number to compare the performance between them.




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
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Algorithm 1</th>
      <td>0.50</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>Algorithm 2</th>
      <td>0.70</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>Algorithm 3</th>
      <td>0.02</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



We could simply take the average of precision and recall $\left(\frac{P+R}{2}\right)$, but this would not be a good strategy for extremes values of the scores. Suppose we have an algorithm predicting $y=1$ all the time (like algorithm 3), it would have very low precision but that would be balanced by a very high recall and it would come out as the best of the three algorithms.




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
      <th>Precision</th>
      <th>Recall</th>
      <th>Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Algorithm 1</th>
      <td>0.50</td>
      <td>0.4</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>Algorithm 2</th>
      <td>0.70</td>
      <td>0.1</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>Algorithm 3</th>
      <td>0.02</td>
      <td>1.0</td>
      <td>0.51</td>
    </tr>
  </tbody>
</table>
</div>



A different way to combine precision and recall is the $F_1$ Score is the harmonic mean between precision and recall $\left(2\frac{PR}{P+R}\right)$ and it's very sensitive to extreme values of precision and recall so that if either one of them is $\approx 0$, also $F_1$ Score will be $\approx 0$




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
      <th>Precision</th>
      <th>Recall</th>
      <th>Average</th>
      <th>$F_1$ Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Algorithm 1</th>
      <td>0.50</td>
      <td>0.4</td>
      <td>0.45</td>
      <td>0.444</td>
    </tr>
    <tr>
      <th>Algorithm 2</th>
      <td>0.70</td>
      <td>0.1</td>
      <td>0.40</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th>Algorithm 3</th>
      <td>0.02</td>
      <td>1.0</td>
      <td>0.51</td>
      <td>0.039</td>
    </tr>
  </tbody>
</table>
</div>



When measuring performance of a learning algorithm at different thresholds, you should use the **cross validation set** to pick the threshold that maximizes $F_1$ Score (if that's the optimization metric of your choice).
