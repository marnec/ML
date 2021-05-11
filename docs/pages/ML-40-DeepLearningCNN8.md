---
layout: default
title: "Deep Learning - CNN - Face verification"
categories: deeplearning
permalink: /ML40/
order: 40
comments: true
---

# Face Verification
Face recognition together with liveness detection is a technology that is increasingly used for authentication in different environments. In face recognition literature there is a distinction between **face verification** and **face recognition**. In the face verification problem the algorithm is given an input image as well as a Name or identification support of a person and the task is to verify whether the input image is that of the claimed person. This kind of problem is sometimes referred to as $1:1$ problem, where the algorithm compare one input picture with the store image associated to the claimed person. The face recognition problem is much harder as it needs to compare the input picture against a database of K persons and output the ID of the matching person in DB (if any). This is hence a $1:K$ problem as you need to compare one input picture with $K$ images.

## One-shot learning
The reason why face verification is a hard task is because it is what is called a one-shot learning problem. In other words your learning algorithm needs to recognize a person given just one example of the person's face, given an an input picture that is different each time and that can vary wildly (<a href="#fig:lightcond">Figure 118</a>).


<figure id="fig:lightcond">
    <img src="{{site.baseurl}}/pages/./data/img/lightconditions.png" alt="png">
    <figcaption>Figure 118. Effect of different light source directions on the pictures of the face of a person taken from the same exact angle and distance</figcaption>
</figure>

Building a system where each person is associated to a class and a classifier tries to match an input picture to the correct class would lead inevitably to failure since you only have one train example for each class. Furthermore, if a new person is added to the people that need to be verified, the algorithm would need to be retrained.

One-shot learning works instead by **learning a similarity function** that outputs the degree of difference $d$ between two input images. Ideally, for the same person we would like $d$ to be very small, while for two different people we would like $d$ to be large. For values of $d$ below a certain threshold $\tau$ (an hyperparameter) we would predict that the two input pictures belong to the same person, while for $d > \tau$, we would predict that the pictures belong to two different people.

$$
d(\small\text{img1},\small\text{img2})
$$

So a face verification algorithm works by taking as input a picture comparing it with all pictures in the database of verified people. For a person that is in the database, we expect that one comparison will return a small value of $d$. For a person that is not in the database, all comparison will hopefully return large values of $d$.

## Siamese network
An implementation of a network to compute the difference between two images is the **siamese network**. The basic compoenent of a siamese network is a clsasical processing of an image through some early convolutional layers and late fully connected layers. Sometimes the last hidden layer is fed to a softmax classifier, but in this case we are instead interested in the last hidden layer itself. In a typical siamese network this is composed of 128 units and encodes for information in the the input picture. Let's call this vector of 128 elements $f(x^{(1)})$ and think of it as an encoding of picture $x^{(1)}$ that represents the input image as a vector of 128 numbers. In a siamese neural network, different pictures are fed to the same neural network with the same weights in order to obtain their encoding in 128-dimensional vectors. Let's say that we input picture $x^{(2)}$ to the network and obtain $f(x^{(2)})$ (<a href="#fig:siamesenet">Figure 119</a>).


    

<figure id="fig:siamesenet">
    <img src="{{site.baseurl}}/pages/ML-40-DeepLearningCNN8_files/ML-40-DeepLearningCNN8_3_0.svg" alt="png">
    <figcaption>Figure 119. siamese network</figcaption>
</figure>

Given $x^{(1)}, x^{(2)}, d(x^{(1)}), f(x^{(2)})$ we can define the distance between $^{(1)}$ and $x^{(2)}$ as the norm of the difference between the encodings of these two images.

$$
d(x^{(1)}, x^{(2)})=\left \| f(x^{(1)}) - f(x^{(2)})  \right \|_2^2
$$

The idea of running two identical CONV networks on two different inputs and compare their output is sometimes called **siamese network architecture** from [this research article](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf).

More formally and succinctly, the idea of a siamese network for face verification is that:

* The network is trained so that its parameters define an encoding $f(x^{i}) \in \mathbb{R}^{128}$
* The parameters are learned so that:
    * if $x^{(i)},x^{(j)}$ are the same person, $\left \| f(x^{(1)}) - f(x^{(2)})  \right \|_2^2$ is small
    * if $x^{(i)},x^{(j)}$ are different persons, $\left \| f(x^{(1)}) - f(x^{(2)})  \right \|_2^2$ is large

## Triplet loss
In order to train a network to produce a good enconding of a person face the **triplet loss** function works particularly well. Our learning objective is to compute the distance between two images: the picture that will identify one person in the final algorithm (also called the anchor) and an input picture. If the input picture is a positive example (belongs to the same person as the anchor) the distance needs to be small. If the input picture is a negative example the distance needs to be large. The reason why this.

$$
\begin{split}
& \underbrace{\left \| f(A) - f(P)  \right \|^2}_{d(A, P)} \leq \underbrace{\left \| f(A) - f(N)  \right \|^2}_{d(A, N)}\\
& \left \| f(A) - f(P)  \right \|^2 - \left \| f(A) - f(N)  \right \|^2 \leq 0 \\
\end{split}
$$

A trivial (but obviously undesired) way to satisfy this equation is if all images have identical encoding $f(x^{i})=k$, so we would have $(k-k)^2-(k-k)^2=0$. In order to prevent the network from learning that all images have identical encoding we need to modify the objective to

$$
 \left \| f(A) - f(P)  \right \|^2 - \left \| f(A) - f(N)  \right \|^2 \leq 0 - \alpha
$$

Which is usually found in the form

$$
\begin{equation}
 \left \| f(A) - f(P)  \right \|^2 - \left \| f(A) - f(N)  \right \|^2 + \alpha \leq 0 
\end{equation}
\label{eq:tripletlosscond} \tag{1}
$$

Where $\alpha$ is an hyperparameter called **margin** (reminiscent of SVM) that is used to prevent the network from learning trivial solutions by pushing further away the difference between $d(A,P)$ and $d(A,N)$.

More formally, the triple loss function is defined on triple on images: given three images $A, P$ and $N$ we can define the loss $\mathcal{L}$ as


$$
\mathcal{L}(A, P, N) = \max \left(\left \| f(A) - f(P)  \right \|^2 - \left \| f(A) - f(N)  \right \|^2 + \alpha, 0 \right)
$$

so that if the distance is $\leq 0$, than the loss $\mathcal{L}=0$, while if the distance is $> 0$, than the loss $\mathcal{L}>0$. The overall cost function $J$ for the neural network is 

$$
J = \sum_{i=1}^m \mathcal{L}\left(A^{(i)}, P^{(i)}, N^{(i)}\right)
$$

### Choosing the triplets A,P,N
For training the network the triplet loss function always require 3 pictures ($A, P, N$) and since A and P are pictures of the same person, this implies that the training set needs multiple pictures of the same person. For example, we could have a training set of 10,000 pictures of 1,000 people. To train the face verification model, we would need to select triplets from the training set to compute the triplet loss function.

Choosing the triplets from the training set is an important step in the training of our one-shot algorithm: If at training the triplets $A,P,N$ are chosen randomly, then the condition $\eqref{eq:tripletlosscond}$ (given that the picture enconding is reliable) is easily satisfied, thus the algorithm will not learn much from them. 

So in order to train the algorithm correctly, we should choose triplets that are hard to train on, in other words we should choose triplets for which $d(A, P) \approx d(A, N)$. If the triplets are chosen randomly, the gradient descent might take too long to get to acceptable performance, as shown in [this article](https://arxiv.org/abs/1503.03832).

Commercial face verification algorithms are trained on fairly large datasets, on the order of a million to 10 million images, with some talk to train on 100 million images.

## Face verification as a binary classification problem
Interestingly, [the article proposing the siamese network](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf), engineers similarity function learning as a binary classification problem instead that a triplet loss optimization problem.

Two identical networks taking part in a siamese network encode two different images and then merge in a logistic regression output unit with target output 1 if the two input images belong to the same person, and 0 if they belong to two different persons <a href="#fig:faceverbin">Figure 120</a>.


    

<figure id="fig:faceverbin">
    <img src="{{site.baseurl}}/pages/ML-40-DeepLearningCNN8_files/ML-40-DeepLearningCNN8_8_0.svg" alt="png">
    <figcaption>Figure 120. Face verification as a binary classification problem. A siamese network merge in a logistic regression unit that produces a binary output: 1 if the two inputs belong to the same person and 0 otherwise.</figcaption>
</figure>

In a binary classification implementation of the siamese network $\hat{y}$ is a non-linearity function applied to the encoding of the images

$$
\hat{y} = \sigma \left( \sum_{k=1}^{128} w_k S  +b \right)
$$

where $w_i$, $b$ are the trained parameters of the 128 features of the encodings to predict if the input pictures are of the same person or not and $S$ in the similarity function

$$
\begin{equation}
S=\left | f \left(x^{(i)} \right)_k - f \left(x^{(j)} \right)_k \right |
\end{equation}
\label{eq:binarysiamese} \tag{2}
$$

However $\eqref{eq:binarysiamese}$ is not the only way to compare the encodings, another possible way is by using the so called $\chi^2$ similarity

$$
S=\frac{\left (f \left(x^{(i)} \right)_k - f \left(x^{(j)} \right)_k \right)^2}{f \left(x^{(i)} \right)_k + f \left(x^{(j)} \right)_k}
$$
