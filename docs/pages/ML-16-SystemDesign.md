---
layout: default
title: "System design - numerical evaluation"
categories: design
permalink: /ML16/
order: 16
comments: true
---

# Machine learning system design
What are the decision that we have to take to effectively and efficiently train a machine learning algorithm? Suppose we want to build a classifier that can discriminate between spam and legitimate emails.

How do we choose the features for our emails?

There is not really a way to tell in advance if you'll need complex features, more data or anything else so a good approach is usually to take these 3 steps:

* start with a simple algorithm that can be implemented quickly, implement it and test it on the cross-validation set;
* plot learning curves to decide what is more likely to help;
* error analysis: manually examine the examples in the cross validation set that your algorithm made errors on: identifying possible systematic errors can help to detect current shortcomings of the system.

Let's say that you have built spam-email classifier and you have $m_\text{CV}=500$ examples in the cross validation set and the algorithm misclassifies 100 emails. Following the suggestions above you would manually examine the 100 misclassified email and categorize them based on:

* type of email
* what feature(s) would have helped correctly classify the email

What we want to achieve is understanding what examples our algorithm finds difficult and what features is worth to spend time on. Very often different algorithm will have difficulties on the same set of examples so starting from a quick and simple algorithm may save you quite a lot of time and pointing you in good directions about what is worth prioritizing.

## The importance of numerical evaluation
When evaluating a learning algorithm it is very helpful to have a proxy of the performance of your algorithm in a single number.

Suppose you have to decide if including [stemming](https://en.wikipedia.org/wiki/Stemming) as a feature of your spam-email classifier. manually looking at wrong examples in the cross validation set may not be a good way to determine if including stemming improves performance. By looking at the percentage of errors on the cross-validation set will instead give you immediately an idea of how much a certain operation impacts on the algorithm performance.

Sometimes looking at the errors is not a good proxy of performance of a learning algorithm and more sophisticated measures are used. This will be expanded upon on later articles.
