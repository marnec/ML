---
layout: default
title: "Deep Learning - CNN - YOLO"
categories: deeplearning
permalink: /ML38/
order: 38
comments: true
---

# YOLO
While the convolutional implementation of the sliding window is more computationally efficient that running a CNN independently for each step of a classic sliding window approach, it has the problem of outputting inaccurate bounding boxes. Suppose our convolutional localization algorithm takes as input a series of windows but none of them really matches ground truth (A) and maybe the best one (B) is as the one in <a href="#fig:bboxes">Figure 106</a>. 


    

<figure id="fig:bboxes">
    <img src="{{site.baseurl}}/pages/ML-38-DeepLearningCNN6_files/ML-38-DeepLearningCNN6_2_0.svg" alt="png">
    <figcaption>Figure 106. Labelled (A) and predicted (B) bounding boxes localizing a car in a picture</figcaption>
</figure>

The [**YOLO**](https://arxiv.org/abs/1506.02640) algorithm allows to increase the accuracy of predicted bounding boxes. The YOLO (**You Only Look Once**) algorithm functions by applying a grid to the input image. In <a href="#fig:yologrid">Figure 107</a> we divide the input image in a $4 \times 4$ grid although in an actual implementation a finer grid is usually employed as for example a $19 \times 19$ grid. The basic idea of the YOLO algorithm is to apply the image classification and localization algorithm seen before and apply it to each cell in the grid.


    

<figure id="fig:yologrid">
    <img src="{{site.baseurl}}/pages/ML-38-DeepLearningCNN6_files/ML-38-DeepLearningCNN6_4_0.svg" alt="png">
    <figcaption>Figure 107. yolo</figcaption>
</figure>

## Intersection over union
Intersection over union can be used to evaluate an object detection algorithm and it is also instrumental to **nonmax suppression**. Suppose you have dataset of pictures labelled with the location of some objects (e.g. cars). Suppose that you develop a CNN that localizes cars trained on that dataset and once run, you have a ground truth bounding box (A) and a predicted bounding box (B) as in <a href="#fig:bboxes">Figure 106</a>. How do you compare a predicted bounding box (B) against the ground truth (A)?

To evaluate bounding boxes we use an index called **intersection over union**, which is a measure of the overlap between two bounding boxes:

$$
\text{IoU} = \frac{A \cap B}{A \cup B} 
$$

By convention many localization tasks will evaluate an answer as correct if $\text{IoU} \geq 0.5$.
