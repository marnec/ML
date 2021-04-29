---
layout: default
title: "Deep Learning - CNN - YOLO"
categories: deeplearning
permalink: /ML38/
order: 38
comments: true
---

# YOLO
While the convolutional implementation of the sliding window is more computationally efficient that running a CNN independently for each step of a classic sliding window approach, it has the problem of outputting inaccurate bounding boxes. Suppose our convolutional localization algorithm takes as input a series of windows but none of them really matches ground truth (A) and maybe the best one (B) is as the one in <a href="#fig:bboxes">Figure 106</a>. The [**YOLO**](https://arxiv.org/abs/1506.02640) algorithm allows to increase the accuracy of predicted bounding boxes. 


    

<figure id="fig:bboxes">
    <img src="{{site.baseurl}}/pages/ML-38-DeepLearningCNN6_files/ML-38-DeepLearningCNN6_2_0.svg" alt="png">
    <figcaption>Figure 106. Labelled (A) and predicted (B) bounding boxes localizing a car in a picture</figcaption>
</figure>

The YOLO (**You Only Look Once**) algorithm functions by applying a grid to the input image. In <a href="#fig:yologrid">Figure 107</a> we divide the input image in a $4 \times 4$ grid although in an actual implementation a finer grid is usually employed as for example a $19 \times 19$ grid. The basic idea of the YOLO algorithm is to apply the image classification and localization algorithm seen before and apply it to each cell in the grid. 


    

<figure id="fig:yologrid">
    <img src="{{site.baseurl}}/pages/ML-38-DeepLearningCNN6_files/ML-38-DeepLearningCNN6_4_0.svg" alt="png">
    <figcaption>Figure 107. yolo</figcaption>
</figure>

As we have seen for the localization object detection algorithm, the label vector $y$ for each grid cell for training the YOLO algorithm will be

$$
y = 
\begin{bmatrix} 
P_c \\
b_x \\
b_y \\
b_h \\
b_w \\
c_1 \\
c_2 \\ 
c_3 \\
\end{bmatrix}
$$

where $p_c$ is a label indicating the presence ($p_c=1$) or absence ($p_c=0$) of any object in the grid cell, $b_x$, $b_y$ are the coordinates of the middle-point of the bounding box, $b_h, b_w$ are the height and width of the bounding box and $c_1, c_2, c_3$ are three different object classes (e.g. car, motorcycle pedestrian). An object is assigned to a cell ($p_c=1$) if the midpoint of its bounding box falls in that cell. So, if a cell contains a portion of a bounding box but not its midpoint it is labeled as background ($p_c=0$). For example the label vectors for grid cells in <a href="#fig:yologrid">Figure 107</a> are

$$
y_{0,n} = y_{1, 1} = y_{1, 3} = y_{2,n} = y_{3,n} =
\begin{bmatrix} 
0 \\
? \\
? \\
? \\
? \\
? \\
? \\ 
? \\
\end{bmatrix} \qquad \qquad 
y_{1, 0} = 
\begin{bmatrix} 
1 \\
\color{blue}{b_x \\
b_y \\
b_h \\
b_w \\}
1 \\
0 \\ 
0 \\
\end{bmatrix}
\qquad \qquad
y_{1, 3} = 
\begin{bmatrix} 
1 \\
\color{orange}{b_x \\
b_y \\
b_h \\
b_w \\}
1 \\
0 \\ 
0 \\
\end{bmatrix}
$$ 

## Intersection over union
Intersection over union can be used to evaluate an object detection algorithm and it is also instrumental to **nonmax suppression**. Suppose you have dataset of pictures labelled with the location of some objects (e.g. cars). Suppose that you develop a CNN that localizes cars trained on that dataset and once run, you have a ground truth bounding box (A) and a predicted bounding box (B) as in <a href="#fig:bboxes">Figure 106</a>. How do you compare a predicted bounding box (B) against the ground truth (A)?

To evaluate bounding boxes we use an index called **intersection over union**, which is a measure of the overlap between two bounding boxes:

$$
\text{IoU} = \frac{A \cap B}{A \cup B} 
$$

By convention many localization tasks will evaluate an answer as correct if $\text{IoU} \geq 0.5$.
