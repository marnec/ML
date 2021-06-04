---
layout: default
title: "Deep Learning - RNN - Sequence to sequence models"
categories: deeplearning
permalink: /ML46/
order: 46
comments: true
---

# Sequence to sequence models
Sequence to sequence models are RNN algorithms that take sequences as input and produce sequences as output. They are employed in many fields from machine translation to speech recognition.

Machine translation is the perfect example of the reason why sequence to sequence models have been developed. Almost all the examples seen so far map an input sequence to a single output. In machine translation we want to translate a sentence from a language to a second sequence in another. 

## Basic seq-to-seq model
Suppose we have a sentence in French that we want to map to its English translation:

$$
\begin{split}
&\begin{array}
&x^{\langle 1 \rangle} & x^{\langle 2 \rangle} & x^{\langle 3 \rangle} & x^{\langle 4 \rangle} & x^{\langle 5 \rangle} \\
\text{Jane} & \text{visite} & \text{l'Afrique} & \text{en} & \text{septembre}
\end{array}\\ \\
&\begin{array}
&\text{Jane} & \text{is} & \text{visiting} & \text{Africa} & \text{in} & \text{September} \\
y^{\langle 1 \rangle} & y^{\langle 2 \rangle} & y^{\langle 3 \rangle} & y^{\langle 4 \rangle} & y^{\langle 5 \rangle} & y^{\langle 6 \rangle} 
\end{array}
\end{split}
$$

In order to map such input sequences to output sequences, [Sutskever et.al. 2014](https://arxiv.org/abs/1409.3215) and [Cho et.al.2014](https://arxiv.org/abs/1406.1078) developed RNN based seq-to-seq models. These models are made of an **encoder** network and a **decoder** network. The encoder processes the input sequence and then feeds it to the decoder, which produces the output sequence (<a href="#fig:seq2seqbasic">Figure 145</a>).


    

<figure id="fig:seq2seqbasic">
    <img src="{{site.baseurl}}/pages/ML-46-DeepLearningRNN4_files/ML-46-DeepLearningRNN4_2_0.svg" alt="png">
    <figcaption>Figure 145. A basic sequence to sequence RNN architecture with an encoder and a decoder. The encoder process the input sequence and feeds it to the decoder, which produces th output sequence</figcaption>
</figure>

The encoder and decoder are two separate networks that serve separate purposes. The task of the encoder is to produce a standard encoding (or we could say an embedding) of an input, while the task of the decoder is to interpret map the encoding to a sequence. 

To further sustain this point, the decoder can also be a convolutional network processing an image. In **image captioning** tasks in fact the ALEX-net, once the final sofmtax layer is removed, plays the role of the decoder network producing a 4096-dimensional encoding of an image, which is then fed to a decoder that produces a caption for the image. 

## Selecting the most likely translation
The decoder network in <a href="#fig:seq2seqbasic">Figure 145</a>) in that it outputs the probability of the translated sentence 

$$P \left(y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle} \right)$$

However, differently from a language model, its input is not a 0-padded vector $a^{\langle 0 \rangle}$ but it is instead the encoding produced by the encoder network. Thus, this language model outputs the conditional probability of the translation, given a certain input sentence $x$

$$
\begin{equation}
P \left (y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle} \vert x \right )
\end{equation} \label{eq:transl} \tag{1}
$$

Each, input sentence might have different output sentences, each with its own probability associated. For example for the input french sentence:

```
Jane visite l'Afrique en septmebre
```

We could have the following sentences:

```
Jane is visiting Africa in September

Jane is going to be visiting Africa in September

In September, Jane will visit Africa

Her African friend welcomed Jane in September
```

So $\eqref{eq:transl}$ denotes a distribution of probabilities associated to different sentences. When feeding the input sentence to the machine translation algorithm we don't want to sample randomly from the output distribution since, while all sentences are strictly speaking correct, we can easily say that some are better translations than others in most contexts. Instead we want to select the **most likely translation**.

$$
\begin{equation}
\underset{y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle}}{\arg \max} P \left (y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle} \vert x \right )
\end{equation} \label{eq:mostliktransl} \tag{2}
$$

### Greedy Search
One possible, suboptimal, to select the most likely translation is to perform a **greedy search**. In a greedy search we would proceed to iteratively select the most likely element in the sequence (next word in the sentence) until completion.

However, we can see how this approach is suboptimal by looking at the following example. Given the two translations

```
Jane is visiting Africa in September

Jane is going to be visiting Africa in September
```

Where the first is better (more likely) than the second, a greedy search would probably select the second since the words `Jane is going` are very common english words and are more likely than `Jane is visiting`.

### Approximate Search
Instead of maximizing the probability $\eqref{eq:transl}$ by selecting one word at the time, we would like to select the group of words that, together, maximizes that probability.

However to explore the translation sentence space exahustively would the computationally impossible. In fact, given a vocabulary of 10000 words and only considering sentences of 10 words we would have $10^{10000}$ possible sentences to test.

For this reason **approximate search** algorithms are used that find near-optimal solutions quickly. One of the most common approximate search algorithsm is called **Beam search**

## Beam Search
Beam search scans the distribution in $\eqref{eq:transl}$ and tries to select the most likely sentence. in Beam search we start by finding the best first word, but instead of selecting one word (as in a greedy search), we select multiple most likely words for position $\langle 1 \rangle$. 

$$
P \left ( y^{\langle 1 \rangle} \vert x \right)
$$

The number of first words to pick is determined by an hyperparameter called **beam width** ($B$). At each step of Beam search, a number of possible options equal to the beam width are selected. A beam width of 3 means that the 3 most likely first words are used to build a sentence.

Once the array of first words is selected, for each of them the whole vocabulary is scanned to find the most likely following words, given the input $x$ and the first word $y^{\langle 1 \rangle}$

$$
\begin{equation}
P \left ( y^{\langle 1 \rangle},y^{\langle 2 \rangle} \vert x \right) = P \left ( y^{\langle 1 \rangle} \vert x \right)P \left ( y^{\langle 2 \rangle} \vert x,  y^{\langle 1 \rangle} \right)
\end{equation} \label{eq:beam2} \tag{3}
$$

Supposing $B=3$ and vocabulary size $n=10000$, we would have to compute $\eqref{eq:beam2}$ 30000 times. During this second step, exactly $B$ most likely sentences are kept and brought forward. Any sentence whose first word is one of the second words in the $B$ selected sentences, is discarded.

Beam search then proceeds by computing the third word in the translation $y^{\langle 3 \rangle}$ given its first two words and the input.

$$
P \left (  y^{\langle 3 \rangle} \vert x , y^{\langle 1 \rangle}, y^{\langle 2 \rangle} \right )
$$

$B$ options are selected and the process is repeated until the sequence is complete. In the case of a machine translation algorithm, this usually happens when a $\small\text{<EOF>}$ token is reached. The entire process of Beam search can be written as

$$
\arg \max_y \prod_{t=1}^{T_y} P \left( y^{\langle t \rangle} \vert x, y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle} \right)
$$

### Length normalization
Length normalization applied to Beam search will produce much better results.
