---
layout: default
title: "Deep Learning - RNN architectures"
categories: deeplearning
permalink: /ML44/
order: 44
comments: true
---

# RNN architectures
In <a href="{{site.basurl}}/ML/ML43ML-43">ML43</a> we have seen an RNN architecture where $T_x = T_y$. For most applications, this is not the case and the basic RNN architecture that we have seen in <a href="{{site.basurl}}/ML/ML43ML-43">ML43</a> can be modified to map sequence-related $x$ and $y$ with different shapes.

## IO relationship
The basic RNN architecture that we have seen in <a href="{{site.basurl}}/ML/ML43ML-43">ML43</a> maps many input to as many outputs and it is therefore called a Many-to-Many architecture. Furthermore, in that architecture $T_x = T_y$. This condition is not always the case and other relationship balances exist: additionally to the many-to-may we have one-to-one, one-to-many and many-to-one.

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
Suppose we want to create a music generation algorithm, where the goal is for the RNN to output a set of notes from a single number input, representing for example the music genre. One-to-many RNN, also called **sequence generation** algorithms, usually feed the output of a time-step $\hat{y}^{\langle t \rangle}$ to the next time-step (<a href="#fig:otm">Figure 128</a>). Sequence generative models have some more subtlety to them that are better explained in a dedicated section.


    

<figure id="fig:otm">
    <img src="{{site.baseurl}}/pages/ML-44-DeepLearningRNN2_files/ML-44-DeepLearningRNN2_7_0.svg" alt="png">
    <figcaption>Figure 128. One to Many RNN: a single (or even no) input elicit the production of a series of outputs</figcaption>
</figure>

### Many-to-Many
We have already seen a Many-to-Many RNN in <a href="{{site.basurl}}/ML/ML43ML-43">ML43</a>, which we used as a basic example of RNN architecture (<a href="fig:mtm">figure below</a>, panel A). Another case exists of many to many architecture, where $T_x \neq T_y$ (<a href="fig:mtm">figure below</a>, panel B). In this architecture input and output are temporally separated: a series of time-steps $[1, T_x]$ only take inputs and another series of time-steps $[1, T_y]$ only produce outputs. Since there is this clear distinction from input and output time-steps they are also referred to as the **encoder** and **decoder**, respectively: the encoder encodes the input and decoder maps the encoded input to the output.


    

<figure id="fig:mtm">
    <img src="{{site.baseurl}}/pages/ML-44-DeepLearningRNN2_files/ML-44-DeepLearningRNN2_9_0.svg" alt="png">
    <figcaption>Figure 129. Mnay-to-Many RNN architectures. A many to many architecture can map an equal number of input and output (A) or input and output of different sizes (B)</figcaption>
</figure>

## Language models
Language models are one of the most fundamental tasks in natural language processing (NLP) and one where RNNs excel. A language model encode the context of a word in a sentence. Suppose we have a speech-recognition algorithm listening to either one of these two sentences:

```
The apple and pair salad
The apple and pear salad
```

`Pair` and `Pear` are pronounced almost the same but one is much more likely than the other. We can easily understand which one because we consider the context of this word in the sentence. A good speech-recognition algorithm will also be able to discriminate which one is more likely thanks to a **language model**.

A language model outputs the probability of a particular sentence to belong to a language and in doing that, it is a fundamental component of both speech recognition and machine translation algorithms. So for example we would have

$$
\begin{split}
& P(\small{\text{The apple and pair salad}}) = 10 ^{-13} \\
& P(\small{\text{The apple and pear salad}}) = 10^{-10}
\end{split}
$$

In general we would say that a language model outputs the probability $P$ of a sentence, where the sentence is a particular sequence of words $y^{\langle t \rangle}$ (in language models inputs are usually represented with $y$ rather than with the traditional $x$):

$$
P \left(y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, \dots y^{\langle T_y \rangle} \right)
$$


### RNN for language modeling
To build a language model it is necessary a large corpus of text (in the target language) to use as a training set. The corpus is made of sentences as for example

```
Cats average 15 hours of sleep a day
```

Each sentence in the training set is tokenized in words. In tokenization each word is mapped to a vocabulary and represented by a vector $y^{\langle t \rangle}$, which encodes for the position of the token in the vocabulary. Words that are not in the vocabulary are assigned a special token $\langle \text{UNK} \rangle$, which is the vocabulary term for unknown words. Usually an additional token marking the end of a sentence $\langle \text{EOS} \rangle$ is used but is is not essential.

```
Cats average 15 hours of sleep a day <EOS>
```

Tokens are then fed to the RNN one step at a time: 

1. At step 1, the RNN is fed $a^{\langle 0 \rangle}$ and $x^{\langle 1 \rangle}$, which are usually both set to an arrays of zeros. in the first step the RNN is equivalent to a softmax classifier that outputs $y^{\langle 1 \rangle}=P(\text{Cats})$: the word `Cats` mapped in the vocabulary 
2. At step 2, the RNN is fed $y^{\langle 1 \rangle}$ and outputs $y^{\langle 1 \rangle} = P(\text{average} \mid \text{ Cats})$: the conditional probability of the word `average` given `Cats`
3. At step 3, the RNN is fed $y^{\langle 2 \rangle}$ and outputs $y^{\langle 1 \rangle} = P(15 \mid P(\text{average} \mid \text{ Cats}))$: the conditional probability of the word `15` given `Cats average`
4. ...

This process continues up to the EOS step, which computes the probability of ending the sentence given the sentence (<a href="#fig:langmod">Figure 130</a>). By feeding each step the distribution of preceding words this RNN learns to predict one word a time.


    

<figure id="fig:langmod">
    <img src="{{site.baseurl}}/pages/ML-44-DeepLearningRNN2_files/ML-44-DeepLearningRNN2_12_0.svg" alt="png">
    <figcaption>Figure 130. Processing of one training example from a corpus of sentence during training of a language modelling RNN</figcaption>
</figure>

To train this RNN the loss ($\mathcal{L}$) and cost ($J$) functions are defined as:

$$\begin{split}
&\mathcal{L}\left(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle} \right ) = - \sum_i y_i^{\langle t \rangle} \log \hat{y}_i^{\langle t \rangle} \\
&J=\sum_t \mathcal{L}^{\langle t \rangle} \left(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}\right )
\end{split}
$$

### Sampling novel sequences
Once a sequence model is trained you can get a sense of what it has learned by letting it **sample novel sequences**. Let's see this applied to out language model.

A sequence model models the chance of any particular sequence of elements as $P \left(y^{\langle 1 \rangle}, \dots, y^{\langle T_x \rangle} \right)$. This distribution can be sampled to generate novel sequences. The sampling of novel sequences follow the same identical process shown in <a href="#fig:langmod">Figure 130</a>, but in sequence sampling each output $\hat{y}^{\langle t \rangle}$ is randomly sampled from the trained probability distribution of the elements in the training set. 

1. The first sample $\hat{y}^{\langle 1 \rangle}$ is a random sampling of $P \left(y^{\langle 1 \rangle}, \dots, y^{\langle T_x \rangle} \right)$
2. The second sample $\hat{y}^{\langle 2 \rangle}$  is a random sampling from $P \left(?\mid \hat{y}^{\langle 1 \rangle} \right)$
3. ...

The sequence sampling can stop when a EOS is sampled or at a specific number of sampled tokens.

This sequence sample model example is built on a word-level language model but these principle applies to other applications of sequence generative models. For example, we could also build a character-level sampling model (based on a language model trained at character-level) which would sample from a distribution of probability of occurrence of characters instead of words. Such a model would have the advantage of not having to deal with unknown words, but on the other hand it would not be able to capture the relationship between distant ends of a sentence as well as a word-level language model. These distant relationships, or **long-term dependencies**, which tend to be very important in natural language (as well as in many other sequence models), while being better captured by a word-level model, still pose a problem for the simple RNN architecture that we have seen so far.

## Long-term dependencies
Long-term dependencies tend to be very important in many sequence models:
