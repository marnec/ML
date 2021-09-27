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

In order to map such input sequences to output sequences, [Sutskever et.al. 2014](https://arxiv.org/abs/1409.3215) and [Cho et.al.2014](https://arxiv.org/abs/1406.1078) developed RNN based seq-to-seq models. These models are made of an **encoder** network and a **decoder** network. The encoder processes the input sequence and then feeds it to the decoder, which produces the output sequence (<a href="#fig:seq2seqbasic">figure below</a>).


    
![svg](ML-46-DeepLearningRNN4_files/ML-46-DeepLearningRNN4_2_0.svg)
    


<i id="fig:seq2seqbasic">A basic sequence to sequence RNN architecture with an encoder and a decoder. The encoder process the input sequence and feeds it to the decoder, which produces th output sequence</i>

The encoder and decoder are two separate networks that serve separate purposes. The task of the encoder is to produce a standard encoding (or we could say an embedding) of an input, while the task of the decoder is to interpret map the encoding to a sequence. 

To further sustain this point, the decoder can also be a convolutional network processing an image. In **image captioning** tasks in fact the ALEX-net, once the final sofmtax layer is removed, plays the role of the decoder network producing a 4096-dimensional encoding of an image, which is then fed to a decoder that produces a caption for the image. 

## Selecting the most likely translation
The decoder network in <a href="#fig:seq2seqbasic">the figure above</a> is a language model (<a href="page:ML44">ML-44</a>) in that it outputs the probability of the translated sentence 

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
Beam search is an heuristic algorithm that scans the distribution in $\eqref{eq:transl}$ and tries to select the most likely sentence. Differently from exact search algorithms like [BFS](https://en.wikipedia.org/wiki/Breadth-first_search) and [DFS](https://en.wikipedia.org/wiki/Depth-first_search), Beam search is not guaranteed to find the best $P(y \vert x)$ .In Beam search we start by finding the best first word, but instead of selecting one word (as in a greedy search), we select multiple most likely words for position $\langle 1 \rangle$. 

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
P \left ( y^{\langle 1 \rangle},y^{\langle 2 \rangle}, y^{\langle 3 \rangle} \vert x \right) =  P \left ( y^{\langle 1 \rangle} \vert x \right)P \left ( y^{\langle 2 \rangle} \vert x,  y^{\langle 1 \rangle} \right) P \left (  y^{\langle 3 \rangle} \vert x , y^{\langle 1 \rangle}, y^{\langle 2 \rangle} \right )
$$

$B$ options are selected and the process is repeated until the sequence is complete. In the case of a machine translation algorithm, this usually happens when a $\small \langle \text{EOF} \rangle$ token is reached.
At the end of the process the probability to maximize is

$$
P \left ( y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle} \vert x \right) =  P \left ( y^{\langle 1 \rangle} \vert x \right)P \left ( y^{\langle 2 \rangle} \vert x,  y^{\langle 1 \rangle} \right) \dots P \left (  y^{\langle T_y \rangle} \vert x , y^{\langle 1 \rangle}, \dots, y^{\langle T_y-1 \rangle} \right )
$$


which can be written as

$$
\begin{equation}
\arg \max_y \prod_{t=1}^{T_y} P \left( y^{\langle t \rangle} \vert x, y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle} \right)
\end{equation} \label{eq:beamsearch} \tag{4}
$$

### Log sum
Given a large enough $T_y$ or small enough $P \left( y \vert x \right)$, $\eqref{eq:beamsearch}$ can produce extremely small numbers that can cause numerical underflow.

So instead of using $\eqref{eq:beamsearch}$ it is common to use its $\log$. Since the log of a product is equivalent to the sum of the logs of its factors we have

$$
\begin{equation}
\arg \max_y \sum_{t=1}^{T_y} \log P \left( y^{\langle t \rangle} \vert x, y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle} \right)
\end{equation} \label{eq:logbeamsearch} \tag{5}
$$

$\eqref{eq:logbeamsearch}$ is numerically stable and less prone to numerical underflow problems. Since the log function is monotonically increasing (<a href="#fig:log">figure below</a>), we know that maximizing $\eqref{eq:beamsearch}$ should give the same results as maximizing $\eqref{eq:logbeamsearch}$


    
![svg](ML-46-DeepLearningRNN4_files/ML-46-DeepLearningRNN4_7_0.svg)
    


<i id="fig:log">Logarithmic function</i>

### Length normalization
Another side effect of the objective function $\eqref{eq:beamsearch}$ (and $\eqref{eq:logbeamsearch}$) is that it becomes smaller the longer the sequence. This means that it will tend to unnaturally prefer short over long sequences. To prevent this problem the objective function is usually normalized by length by dividing by the number of elements in the output sequence.

$$
\begin{equation}
\frac{1}{T_y^\alpha} \sum_{t=1}^{T_y} \log P \left( y^{\langle t \rangle} \vert x, y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle} \right)
\end{equation} \label{eq:lengthbeamsearch} \tag{6}
$$

Where if the hyperparameter $\alpha=0$, we are maximizing the sum of the logs of probabilities exactly as in $\eqref{eq:logbeamsearch}$ (no normalization), whereas if $\alpha=1$, we are maximizing the average of logs of probabilities (full normalization). Commonly, a value of $\alpha$ in the range $[0, 1]$ is applied to have non-full normalization and a value that has been empirically found to work well for most applications is $\alpha = 0.7$.

### Choice of Beam Width
The larger $B$ is, the more possible sequences we are exploring and thus the better probability of finding the best output sequence. On the other hand, the larger $B$ is, the more computationally expensive the algorithm.

In production systems is not uncommon the find values of $B$ around 10, whereas $B$ around 100 would be considered very large. However, the choice of $B$ is very domain-dependent and, when necessary $B$ can need to be as large as 1000 or 3000.

It is important to notice that in most cases large values of $B$ gives diminishing returns, so it is expected to see huge gains in performance increasing $B$ at low values ($1 \to 10$) while very small gains in performance when increasing $B$ at large values ($1000 \to 3000$)

### Error analysis with Beam Search
Since Beam Search is an heuristic algorithm it doesn't necessarily find an optimal solution. Error analysis of Beam Search can help understand whether it is the Beam Search or the underlying RNN model that needs improving.

Suppose you have the input sentence

```
Jane visite l'Afrique en septembre
```

And the following human translation $y^*$ in your development set

```
Jane visits Africa in September
```

And the output from Beam searching your RNN model $\hat{y}$ is instead

```
Jane visited Africa last September
```

We can't known in *a priori* if the beam width is too small or the RNN model is not performing adequately. Since the RNN is computing $P(y \vert x)$, to understand how well is the model performing we can compare the probability of $y^*$ and $\hat{y}$

$$
P \left(y^* \vert x \right) - P \left(\hat{y} \vert x \right) 
\begin{cases}
> 0 \quad \to \quad \scriptsize\text{improve Beam Search} \\
\leq 0 \quad \to \quad \scriptsize\text{improve RNN}
\end{cases}
$$

* If $P(y^* \vert x) > P(\hat{y} \vert x)$ it means that Beam Search chose $\hat{y}$ but $y^*$ would attain higher $P(y \vert x)$. So Beam Search is failing to find the optimal solution.
* If $P(y^* \vert x) \leq P(\hat{y} \vert x)$ it means that Beam Search chose $\hat{y}$ *because it attains* a higher $P(y \vert x)$. So Beam Search is working fine but the RNN model is not trained well.

To carry out a comprehensive error analysis we can check the responsibility of all errors in our development set. If Beam Search is at fault for a large fraction of them, increasing the value of $B$ should resolve the issue. If errors are due to a faulty RNN model, then we ca carry out a deeper layer of error analysis as detailed in <a href="page:ML15">ML-15</a>.

## BLEU Score
Unlike in tasks like image recognition, where there's one right answer, in machine translation there might be multiple equally good answers and it becomes impossible to measure accuracy traditionally. Conventionally accuracy in machine translation is measured with the **BiLingual Evaluation Undesrstudy (BLEU) score** ([Papineni et.al. 2002](https://www.aclweb.org/anthology/P02-1040.pdf)).

Suppose we have an input French sentence

```
Le chat est sur le tapis
```

and two equally correct human translations

```
The cat is on the mat

There is a cat on the mat
```

The BLEU score works by checking if the type of words in the machine translation appear in at least one of the human generated references. It calculates a modified **precision** score on all **n-grams**, where a unigram is a single word, a bigrmam is a pair of consecutive words and so on.

To understand how the modified precision is calculated, suppose the machine translation output is

```
the the the the the the the
```

To calculate the modified precision of the unigrams, we compute the ratio of the clipped number of occurrences of a unigram ($\scriptsize\text{Count}_{\text{clip}}$) in the machine translation output ($\hat{y}$) to its unclipped occurrences ($\scriptsize\text{Count}$). Clipped occurrences are calculated by counting how many of the unigram in the output appear in any of the references. The count is clipped to the maximum number of times the unigram appears in any one of the references. In this case all $7$ unigrams of the translation appear in the reference but, since the word `the` is found at most 2 times in the references (first reference), the count is clipped to 2 and the precision for unigrams $p_1 = \frac{2}{7}$. In the general case in which there are more than one word in the machine translation, we can write $p_1$ as

$$
p_1 = \frac{\sum_{\small\text{unigram } \in \hat{y}}\small\text{Count}_\text{clip} (\text{unigram})}
{\sum_{\small\text{unigram } \in \hat{y}}\small\text{Count}(\text{unigram})}
$$

And this can be extended to the n-gram case

$$
p_n = \frac{\sum_{\small\text{n-gram } \in \hat{y}}\small\text{Count}_\text{clip} (\text{n-gram})}
{\sum_{\small\text{n-gram } \in \hat{y}}\small\text{Count}(\text{n-gram})}
$$

$p_n = 1$ when the sentence is exactly the same of a reference or a combination of references that hopefully still constitutes a good translation.

The combined BLEU score usually assumes values of $p_n \in [1,4]$ and it is defined as

$$
\text{BP} \exp \left ( \frac{1}{4} \sum_{n=1}^4 p_n \right )
$$

where $\text{BP}$ stands for **brevity penalty** which penalizes translations shorter than the references, which would tend to have bigger scores. $\text{BP}$ is defined as

$$
\text{BP} = 
\begin{cases}
1 &\quad \small \text{len}(\hat{y}) > \text{len}(y^*) \\
\exp \left (1 - \frac{\text{len}(y^*)}{\text{len}(\hat{y})} \right) &\quad \small\text{otherwise} 
\end{cases}
$$
