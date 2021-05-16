---
layout: default
title: "Deep Learning - RNN architectures"
categories: deeplearning
permalink: /ML44/
order: 44
comments: true
---

# RNN architectures
In <a href="{{site.basurl}}/ML/ML43">ML43</a>  can be modified to map sequence-related $x$ and $y$ with different shapes.

## IO relationship
The basic RNN architecture that we have seen in <a href="{{site.basurl}}/ML/ML43">ML43</a>, top-right panel))

### One-to-One
A one-to-one architecture maps an input $x$ to an output $y$ in a single time-step and it is a limit case, which is identical to a classical (non-recurrent) neural network <a href="#fig:rnnarchitectures">Figure 126</a>, top-left panel)

### Many-to-One
Suppose we want to train an RNN for a sentiment classification task where, given an sentence $x$, the task is to produce a number $y$ indicating how good is the sentiment that the sentence expresses. For example, we could have movie reviews as input and the task is to tell if the review is good or bad. The words of the input review are parsed in subsequent time-steps, but only one output is produced at the end of the process (<a href="#fig:rnnarchitectures">Figure 126</a>, bottom-left panel)  This is a Many-to-One architecture where we want to map mulitple inputs (multiple words in a sentence) to a single output (the review mark).

### One-to-Many
Suppose we want to create a music generation algorithm, where the goal is for the RNN to output a set of notes from a single number input, representing for example the music genre. 


    

<figure id="fig:rnnarchitectures">
    <img src="{{site.baseurl}}/pages/ML-44-DeepLearningRNN2_files/ML-44-DeepLearningRNN2_2_0.svg" alt="png">
    <figcaption>Figure 126. different RNN architectures</figcaption>
</figure>
