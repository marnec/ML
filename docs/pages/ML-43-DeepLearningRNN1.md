---
layout: default
title: "Deep Learning - CNN - Sequence models"
categories: deeplearning
permalink: /ML43/
order: 43
comments: true
---

# Sequence Models 
Models like Recurrent Neural Networks (RNNs) have transformed many fields, as for example speech recognition and natural language processing. There are a lot of different types of sequence problems that can be treated as supervised learning problems:

* In speech recognition an input audio source $x$ is mapped to an output text $y$. Both the input and the output are sequence data, because $x$ plays over time and $y$ is a sequence of words.
* In music generation the output $y$ is a sequence, while the input can be void or it may be a single number encoding for the genre or style of music to generated. 
* In sentiment classification the input $x$ is one or more sentences (sequence) while the output $y$ is a number encoding for the quality of the object to which the input refers. 
* Sequence models are also widely used in DNA sequence analysis.
* In machine translation the input $x$ is a sentence that is translated in another output sentence $y$
* In video activity recognition an activity $y$ is recognized from a sequence of video frames $x$
* In name entity recognition entities $y$ like for example peoples are detected in an input sentence $x$

## Notation
Suppose you want a sequence model that extract character names from a sentence $x$ and classifies each word as character name (1) or not (0)

> $x$: ```Rand is a friend of Perrin and they both live in the twin rivers```<br>
$y$:   ```1    0  0 0      0  1      0   0    0    0    0  0   0    0     ```

The input is a sequence of 13 words ($T_x=13$). They are represented by a set of 13 features to which we will refer as $x^{\langle 1 \rangle}, \dots ,x^{\langle 13 \rangle}$ or more in general they will be referred to as $x^{\langle t \rangle}$ where $t$ implies that they are temporal sequences regardless whether the sequence is temporal or not.

Similarly we refer to $y^{\langle 1 \rangle}, \dots , y^{\langle t \rangle} , \dots,y^{\langle 13 \rangle}$ which has length $T_y$. In this example $T_x = T_y = 13$ but they can be different.

Each training example $X^{(i)}$ has a label associated to some features $t$. To refer to feature $t$ of training example $i$ we use $X^{(i)\langle t \rangle}$. Each training example might have a different number of input features $T_x$. To refer to the number of input features of training example $i$ we use $T_x^{(i)}$. Similarly $y^{(i)\langle t \rangle}$ refers to the $t$-th element in the output sequence of the $i$-th training example and $T_y^{(i)}$ refers to the length of the output sequence in the $i$-the training example.


```python

```
