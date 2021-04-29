---
layout: default
title: "Deep Learning - CNN - YOLO"
categories: deeplearning
permalink: /ML38/
order: 38
comments: true
---



## Intersection over union
Intersection over union can be used to evaluate an object detection algorithm and it is also instrumental to **nonmax suppression**. Suppose you have dataset of pictures labelled with the location of some objects (e.g. cars). Suppose that you develop a CNN that localizes cars trained on that dataset and once run, you have a ground truth bounding box (A) and a predicted bounding box (B) as in <a href="#fig:bboxes">Figure 106</a>. How do you compare a predicted bounding box (B) against the ground truth (A)?

To evaluate bounding boxes we use an index called **intersection over union**, which is a measure of the overlap between two bounding boxes:

$$
\text{IoU} = \frac{A \cap B}{A \cup B} 
$$

By convention many localization tasks will evaluate an answer as correct if $\text{IoU} \geq 0.5$.


    

<figure id="fig:bboxes">
    <img src="{{site.baseurl}}/pages/ML-38-DeepLearningCNN6_files/ML-38-DeepLearningCNN6_3_0.svg" alt="png">
    <figcaption>Figure 106. Labelled (A) and predicted (B) bounding boxes localizing a car in a picture</figcaption>
</figure>
