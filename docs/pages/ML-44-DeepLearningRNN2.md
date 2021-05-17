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
The basic RNN architecture that we have seen in <a href="{{site.basurl}}/ML/ML43">ML43</a> maps many input to as many outputs and it is therefore called a Many-to-Many architecture. Furthermore, in that architecture $T_x = T_y$. This condition is not always the case and other relationship balances exist: additionally to the many-to-may we have one-to-one, one-to-many and many-to-one.

### One-to-One
A one-to-one architecture maps an input $x$ to an output $y$ in a single time-step and it is a limit case, which is identical to a classical (non-recurrent) neural network <a href="#fig:oto">Figure 126</a>)


    

<figure id="fig:oto">
    <img src="{{site.baseurl}}/pages/ML-44-DeepLearningRNN2_files/ML-44-DeepLearningRNN2_3_0.svg" alt="png">
    <figcaption>Figure 126. One to One RNN: an RNN with a single time step where the input $x$ passes through the network only once to produce $y$. This is equivalent to a classic (non-recursive) neural network</figcaption>
</figure>

### Many-to-One
Suppose we want to train an RNN for a sentiment classification task where, given an sentence $x$, the task is to produce a number $y$ indicating how good is the sentiment that the sentence expresses. For example, we could have movie reviews as input and the task is to tell if the review is good or bad. The words of the input review are parsed in subsequent time-steps, but only one output is produced at the end of the process (<a href="#fig:mto">Figure 127</a>)  This is a Many-to-One architecture where we want to map mulitple inputs (multiple words in a sentence) to a single output (the review mark).


    

<figure id="fig:mto">
    <img src="{{site.baseurl}}/pages/ML-44-DeepLearningRNN2_files/ML-44-DeepLearningRNN2_5_0.svg" alt="png">
    <figcaption>Figure 127. Many to One RNN: inputs fed through each time-step $x^{\langle t \rangle}$ concur to produce a single output $\hat{y}$</figcaption>
</figure>

### One-to-Many
Suppose we want to create a music generation algorithm, where the goal is for the RNN to output a set of notes from a single number input, representing for example the music genre. One-to-many RNN, also called **sequence generation** algorithms, usually feed the output of a time-step $\hat{y}^{\langle t \rangle}$ to the next time-step (<a href="fig:otm">figure below</a>). Sequence generative models have some more subtlety to them that are better explained in a dedicated section.


    

<figure id="fig:otm">
    <img src="{{site.baseurl}}/pages/ML-44-DeepLearningRNN2_files/ML-44-DeepLearningRNN2_7_0.svg" alt="png">
    <figcaption>Figure 128. One to Many RNN: a single (or even no) input elicit the production of a series of outputs</figcaption>
</figure>

### Many-to-Many
We have already seen a Many-to-Many RNN in <a href="{{site.basurl}}/ML/ML43">ML43</a>, panel B). In this architecture input and output are temporally separated: a series of time-steps $[1, T_x]$ only take inputs and another series of time-steps $[1, T_y]$ only produce outputs. Since there is this clear distinction from input and output time-steps they are also referred to as the **encoder** and **decoder**, respectively: the encoder encodes the input and decoder maps the encoded input to the output.


    

<figure id="fig:mtm">
    <img src="{{site.baseurl}}/pages/ML-44-DeepLearningRNN2_files/ML-44-DeepLearningRNN2_9_0.svg" alt="png">
    <figcaption>Figure 129. Mnay-to-Many RNN architectures. A many to many architecture can map an equal number of input and output (A) or input and output of different sizes (B)</figcaption>
</figure>
