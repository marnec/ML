---
layout: default
title: "Deep Learning - RNN - Attention Model"
categories: deeplearning
permalink: /ML47/
order: 48
comments: true
---

# Attention model 
The attention model is an alternative to the encoder decoder architecture. 

### Attention model motivation
The intuition behind attention models is best introduced by looking at machine translation task for a long sentence. Suppose we have the input sentence

```
Jane s'est rendue en Afrique en septembre dernier, a apprécié la culture et a rencontré beaucoup de gens merveilluex; elle est revenue en parlant comment son voyage était merveilluex, et elle me tente d'y aller aussi.
```

And its reference translation

```
Jane went to Africa last Septembre, and enjoyed the culture and met many wonderful people; she came back raving about how wonderful the trip was, and is tempting me to go too.
```

An encoder-decoder model would take the whole sentence as input, encode it in a $n$-dimensional vector and then pass it to the decoder network to produce a translation. However, this is not how a human translator would proceed. A human translator is likely to translate the sentence one piece a time and proceed until completion. 

And so what we observe for encoder-decoder models is that they work well for short sentences but sentences longer than some tens of words, their performance decay. Attention models don't suffer from this performance decay (<a href="fig:attentionBLEU">figure below</a>).


    
![svg](ML-47-DeepLearningRNN5_files/ML-47-DeepLearningRNN5_2_0.svg)
    


## Attention model
Suppose we have an input sentence

```
Jane visite l'Afrique en septembre
```

And we use a bidirectional RNN (more commonly bidirectional GRU and bidirectional LSTM) to compute features on every word (<a href="#fig:attentionmodel">Figure 147</a>, bottom network). The notation for this bidirectional network is as follows: each step in the input sequence is $t^\prime$,  we denote with $\overrightarrow{a}^{\langle t^\prime \rangle}$ the forward occurrence and with  $\overleftarrow{a}^{\langle t^\prime \rangle}$ the backward occurrence of the network. For brevity we denote 

$$a^{\langle t^\prime \rangle} = \left(\overrightarrow{a}^{\langle t^\prime \rangle}, \overleftarrow{a}^{\langle t^\prime \rangle} \right)$$

Then we have a forward only RNN with state $s^{\langle t \rangle}$ that produces the output sequence $\hat{y}^{\langle t \rangle}$ (<a href="#fig:attentionmodel">Figure 147</a>, top network). Each step is fed a context $c^{\langle t \rangle}$ and the output from the previous step $\hat{y}^{\langle t-1 \rangle}$. The context of each step is fed the output from all the input activations $a^{\langle t^\prime \rangle}$ weighted by the **attention weights** $\alpha^{\langle t, t^\prime \rangle}$. The **attention weights** modulate how much the context of a step in the output sequence depends on the features ($a^{\langle t^\prime \rangle}$) of each time step in the input sequence. The context is in fact a weighted sum of the features of each time step in the input weighted by its attention weight.

$$
c^{\langle t \rangle} = \sum_{t^\prime}\alpha^{\langle t, t^\prime \rangle}a^{\langle t^\prime \rangle}
$$


    

<figure id="fig:attentionmodel">
    <img src="{{site.baseurl}}/pages/ML-47-DeepLearningRNN5_files/ML-47-DeepLearningRNN5_4_0.svg" alt="png">
    <figcaption>Figure 147. An attention model for a machine translation task. The input sequence is processed by a bi-directional network, which feeds into an forward network passing through a context layer. Each context layer is fed the activation (features) of all time-steps of the bi-directional network, weighted by a set of attention weights, which define how much each activation contributes to the output</figcaption>
</figure>

The attention weights satisfy the condition

$$
\begin{equation}
\sum_{t^\prime} \alpha^{\langle t, t^\prime \rangle} = 1
\end{equation} \label{eq:attwond} \tag{4}
$$

This means that the attention weights that modulate the contribution to the output step $t$ of each input steps $t^\prime$, sum to 1. The attention weights definition is built to satisfy $\eqref{eq:attwond}$:

$$
a ^{\langle t, t^\prime \rangle} = \frac{\exp \left( e^{\langle t, t^\prime \rangle} \right)}
{\sum_{t^\prime=1}^{T_x}\exp \left( e^{\langle t, t^\prime \rangle} \right)}
$$

The terms $e^{ \langle t, t^\prime \rangle}$ are usually computed by using a small neural network (usually with 1 hidden layer) that takes as input the state from the previous time step ($s^{\langle t-1 \rangle}$) and the features from the time step $t^\prime$, $a^{\langle t^\prime \rangle}$ (<a href="#fig:attentionweightnetwork">Figure 148</a>). The intuition behind this process is that if you want to know how much the current step $s^{\langle t \rangle}$ should pay to the activation $a^{\langle t^\prime \rangle}$, it should depend on the previous state  $s^{\langle t-1 \rangle}$ and on the activation itself.


    

<figure id="fig:attentionweightnetwork">
    <img src="{{site.baseurl}}/pages/ML-47-DeepLearningRNN5_files/ML-47-DeepLearningRNN5_6_0.svg" alt="png">
    <figcaption>Figure 148. The network that determines the weighting function for state $s^{\langle t \rangle}$ is based on the previous state $s^{\langle t-1 \rangle}$  and on the features at $t^\prime$</figcaption>
</figure>

The neural network that sets the attention weights is trained as part of the whole model, so that the reference translations are themselves responsible of setting the function that determines the attention weights from the previous state in the output $s^{\langle t-1 \rangle}$ and the activation from each input time steps $a^{\langle t^\prime \rangle}$.

The downside of this architecture is that it has a quadratic cost. Given $T_x$ steps in the input and $T_y$ steps in the output, the attention parameters are $T_x \cdot T_y$.

Other than machine translation, the attention model has been applied to image captioning by [Xu et.al.2015](https://arxiv.org/pdf/1502.03044.pdf), who have shown hoe the caption can be generated by focusing (setting the attention) on part of a picture at a time.
