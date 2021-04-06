---
layout: default
title: "Deep Learning - Softmax classifier"
categories: deeplearning
permalink: /ML32/
order: 32
comments: true
---

## Orthogonalization
In training a ML algorithm, it is important to know what to tune in order to achieve a certain effect. A status in which each hyper-parameters tune exactly one aspect of the model is a status of perfect **orthogonalization** and is a status that we would like to achieve.

In machine learning, the effects that you want to orthogonalize are:

1. Fit training set well on cost function (for some applications this means approaching human-level performance): e.g. tune by getting a bigger network
2. Fit dev set well on cost function: e.g. tune by getting a bigger training set
3. Fit test set well on cost function: e.g. tune by getting a bigger dev set
4. Performs well in real world: e.g. tune by changing cost function

An action that doesn't fit well with orthogonalization is early stopping (<a href="{{site.basurl}}/ML/ML27">ML27</a>), since it tries to simultaneously tunes train set and dev set performance.

## Satisficing and Optimizing metrics
We have talked in <a href="{{site.basurl}}/ML/ML17">ML17</a> about the importance of having a single real-number evaluation metric. However, it is not always easy to combine all the desired properties of a model in a single metric. In those cases it is useful to set satisficing and optimizing metrics.

Let's say that for an image classifier we care about the classification accuracy and about the running time. Suppose we have three classifiers as in the <a href="#satopt">the table below</a>




<style  type="text/css" >
</style><table id="T_da798_" id="satopt"><caption>Three classifiers and their accuracies and running times</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >Running time</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_da798_level0_row0" class="row_heading level0 row0" >A</th>
                        <td id="T_da798_row0_col0" class="data row0 col0" >90%</td>
                        <td id="T_da798_row0_col1" class="data row0 col1" >80ms</td>
            </tr>
            <tr>
                        <th id="T_da798_level0_row1" class="row_heading level0 row1" >B</th>
                        <td id="T_da798_row1_col0" class="data row1 col0" >92%</td>
                        <td id="T_da798_row1_col1" class="data row1 col1" >95ms</td>
            </tr>
            <tr>
                        <th id="T_da798_level0_row2" class="row_heading level0 row2" >C</th>
                        <td id="T_da798_row2_col0" class="data row2 col0" >95%</td>
                        <td id="T_da798_row2_col1" class="data row2 col1" >1500ms</td>
            </tr>
    </tbody></table>



We may set some rules that we want models to be subject to:

* maximize accuracy
* running time $\leq 100 \mathrm{ms}$

In this case accuracy is an optimizing metric because we want to get as good as possible on this metric, while running-time is a satisficing metric because we want to have at least a certain running-time. In general, if we decide that we care about $N$ metrics, we should have 1 optimizing metric and $N-1$ satisficing metrics.

##  Human level performance
Deep-learning applications are often compared with human-level performance. First, because ML is become so advanced that can perform as well or better than human-level performance. Second, because it turns out that the model workflow is much more efficient when the algorithm is trying to do something that also humans can do. In those settings, it becomes natural to compare with human-level performance

Typically, when designing a ML model, performance will rapidly increase at first, approach and surpass human level performance and then asymptotically approach an ideal performance, called **Bayes optimal error**, which is the best theoretical function mapping $x \to y$, that can never be surpassed.


    

<figure id="fig:pertrend">
    <img src="{{site.baseurl}}/pages/ML-33-DeepLearningStrategy_files/ML-33-DeepLearningStrategy_6_0.png" alt="png">
    <figcaption>Figure 75. Typical trend of performance of a trained model with human level performance and Bayes optimal performance.</figcaption>
</figure>

Interestingly, usually performance increase, really slows down after surpassing human-level performance. This happens for (at least) two reasons:

* Human level performance is, for many tasks, very close to Bayes optimal error
* So long as ML is worse than humans there are certain things that we can do to improve performance:
    * get labeled data from humans
    * gain insight from manual error analysis (why did a person get this right?)
    * better analysis of bias/variance (<a href="{{site.basurl}}/ML/ML25">ML25</a>)

### Define human level performance
The first step towards understanding the human level error for a certain task is to decide what purpose would serve its definition.

Suppose you are developing a model for the classification of radiology scans, and you find the following typical errors:

1. common person: 3% error
2. common doctor: 1% error
3. experienced radiologist: 0.7% error
4. team of experienced radiologists: 0.5%

If, as often is the case, you are using human level error as a proxy for Bayes error, you would chose option 4 as your human level performance. If you are developing a model for research purposes and you want to demonstrate that the model is usable in real world environment, you would probably chose option 2 as human level performance.

### Manual Error analysis
Suppose you are training a cat classifier, which has 90% accuracy and 10% errors. You notice that some dogs are incorrectly classified as cats. Should you try yo make your cat classifier perform better on dogs?

In order to answer this question it is often advisable to manually look at the data in order to advance an estimate of the maximum gain in performance that you might have by doing specific action; this is called **error analysis**.

In this case, you would extract around 100 mislabeled examples from the dev set and count how many of them are dogs. If 5% of them are dogs, you know that the maximum gain in performance will be to lower you error rate from 10% to 9.5%. However, if you notice that 50% of mislabeled examples are dogs, then you may be able to reduce your error from 10% to 5%.

In this example, we analyzed a single problem (dogs misclassified as cats) but, while you sort through your dev set misclassified examples, you can evaluate multiple options in parallel. For example, you may notice that some misclassified examples are great cats and some others are blurry images.

#### Special case of incorrectly labeled examples
Deep learning algorithms are quite robust to random (or near-random) errors in the training set, they are less robust to systematic errors in the training set. When it comes to incorrectly labeled examples in the dev set, it is often a good idea have a separate count for incorrectly labeled examples during the manual error analysis.

The decision on whether or not to go through your entire dev set and correct the incorrectly labeled examples is bound to the same evaluation as for the other type of misclassifications found in manual error analysis. Since the goal of the dev set is to help you select between different models, if you don't trust anymore the performance evaluated on your dev set, then by all means you should correct misclassifications.

A couple of important caveats are:

* Apply the same process to your dev and test sets to make sure that they continue to come from the same distribution
* Consider examining examples your algorithm gt right as well as ones it got wrong. 
* you don't need to apply mislabeled correction to the train set if you apply it to your dev/test set. Train and dev/test data may now come slightly different distributions.

## Mismatched train and dev/test set
In the deep learning era, the need for data is such that sometimes you will use whatever data you have, regardless if it comes from the same distribution of the test/dev set. Training on a different distribution than dev/test set is possible but requires some considerations.

Suppose we are training a model that needs to classify pictures uploaded by users. These pictures are few in number and typically, they're blurry and shot by unprofessional photographers. On the other hand you have access from the web to a lot of professionally shot, high resolution pictures.

Your final objective is your ML performing well on user-fed pictures and you find yourself in front of a dilemma:

* you have a small dataset with a certain distribution that upon which your model needs to perform well
* you have a large dataset with a different distribution upon which you could train your model

### Option 1: Merge the datasets
One option would be to merge the two datasets, randomly reshuffle them and then split in training dev and test sets.

* Advantage: This approach has the advantage that the train and dev/test set will all come from the same distribution. 
* Disadvantage: The huge disadvantage is that a big portion of the dev set will be composed of the high-quality pictures. This means that model selection (operated on the results of the model on the dev set) will bring to optimize for a different distribution of data from what we care about.

### Option 2: Test on the target dataset
Split your target dataset (user-fed pictures) in two batches in an appropriate portion. One portion will go in the training set, the other portion will go in the dev/test set. All of the high-resolution pictures go in the training set.

* Advantage: You test your model on your target distribution, optimizing exactly on the distribution we care about
* Disadvantage: The training distribution is different from your dev/test distribution, however over the long term this split of data across the subsets usually give better performance.


## Bias and Variance with mismatched data
Suppose we are in the situation in which our training set come from a different distribution than our dev/test set. Let's assume that human level error $\approx 0%$

|                |     |
|----------------|-----|
| Training error | 1%  |
| Dev error      | 10% |

In a setting where train and dev set come from the same distribution, this is a case of high variance. But when train and dev set come from different distributions this is no more obvious. The 9% increase in error might be due to variance, but it could also be due to the different distributions of the train and dev set. In order to decouple this two problems we define an additional dataset, called **training-dev** set, that has the same distribution as the training set, but is not used for training. 

Before training we define 4 datasets: 

* Training set
* Training-dev set (same distribution as the training set but excluded from training)
* Dev set (different distribution than training set)
* Test set (if needed)

Suppose we find:

|                |   Case 1  |   Case 2  |   Case 3  | Case 4  | Case 5 |
|----------------|-----|-----|-|-|-|
| Training error | 1%  |   1%  | 10%  | 10% | 5%  |
| Training-dev error | 9%  |   1.5%  | 11% | 11% | 7%  |
| Dev error      | 10% |   10%  | 12%  |   20% | 4%  |

Let's take Case 1. We observe that the error increases of 8% from training data to training-dev data. Since there is no difference in distribution between the two datasets, and the only difference is that the model hasn't been trained on the training-dev data, we conclude that the model suffer from high variance.

Let's take Case 2. We observe that the error increases of 0.5% from training data to training-dev data. We conclude that we have low variance and the remaining difference in performance needs to be a **data-mismatch problem**.

Let's take Case 3. Remember that human level error (proxy for Bayes error) $\approx 0%$. We observe almost no difference in error between datasets, but a great difference between Bayes error and training error. This is an **Avoidable Bias problem**.

Let's take Case 4. We observe that a great difference between Bayes error and training error **and** a great difference between training-dev error and dev error. We deduce that this model has two issues: An avoidable bias problem and a data-mismatch problem.

Let's take Case 5. We observe that the dev error is smaller than the training error. We deduce that the training data was much more complex than dev data

A more general formulation

|                |   Training distribution  |  Dev distribution  |  
|----------------|-----|-----|
| Human level |  general human level |   ? (specific human level)  |
| Error on examples NN has trained on | training error  |   ? (error on dev distribution data split in train data) |
| Error on examples NN hasn't trained on | training-dev error |   dev/test error  |

We never cited the two columns marked by the question mark, but having data on those two situations can give us interesting information on the next steps to take.


```python

```
