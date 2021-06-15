---
layout: default
title: "Deep Learning - CNN - 1D/3D generalization"
categories: deeplearning
permalink: /ML42/
order: 42
comments: true
---

# 1D and 3D generalization of CNNs
Most of the discussion around ConvNets revolves around images, which can be seen as 2D data. However the CNN principles applie to 1D as well as 3D data.
## 1D data
For example ECG (ElectroCardioGram) data (<a href="#fig:1dconv">figure below</a>, panel B) is one dimensional as it shows a series of measurements along time. So instead of being an $n \times n$ dimensional input it's just an $n$-dimensional input and is convolved with an $f$-dimensional filter in place of an $f \times f$ dimensional filter.

For 1D or sequence data, the set of algorithms that are most employed are **recurrent neural network** (RNN), but in some situations CNNs can be a good if not better alternative for sequence data modelling.


    
![svg](ML-42-DeepLearningCNN10_files/ML-42-DeepLearningCNN10_2_0.svg)
    


<i id="fig:1dconv">A convolution operation on a 2D input (A) and 1D input (B)</i>

## 3D data
CT scans are fundamentally 3-dimensional since a CT scan produces a set of cross-section images of a volume. Suppose we have a 3D volume of $14 \times 14 \times 14$, it can be convolved with an $n_c'$ number of $5 \times 5 \times 5$ filters, to produce a $10 \times 10 \times 10 \times n_c'$ output volume.

A 3D input doesn't necessarily need to represent an actual volume as in the case of CT-scans, it could be for example a movie, where each channel represents a frame. A CNN applied to such data could detect motion or particular type of actions.
