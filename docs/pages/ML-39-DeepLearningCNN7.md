---
layout: default
title: "Deep Learning - CNN - U-Net"
categories: deeplearning
permalink: /ML39/
order: 39
comments: true
---

# U-Net

## Region proposal
In the body of work about object detection an influential idea that has been proposed is **region proposal**. One of the classic region proposal algorithm is called R-CNN.

In a classic sliding window algorithm we would roll a sliding window across the whole input image, starting from the top-left corner all the way to the bottom-right corner. While we have seen that a convolutional implementation of the sliding window exists, it remains the problem that most of the windows analyzed will contain uninteresting data. 

In an R-CNN an **unsupervised segmentation** filter is first applied to the input image to detect different areas on the image (panel B of <a href="#fig:semseg">Figure 111</a>). This segmentation step produces a number of blobs (usually some thousands). Bounding box are draw around each of these blobs and an object detection algorithm tries to detect objects inside the bounding box.


    

<figure id="fig:semseg">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_2_0.svg" alt="png">
    <figcaption>Figure 111. An example of a very precise semantic segmentation (B) achieved using 3D-data from a Lidar and a video feed from a moving car (A). In a classic sliding window approach (convolutional or not) most of the windows will contain uninteresting data (red bounding box), while only some areas will actually contain an object (blue bounding box)</figcaption>
</figure>

This method hugely reduces the number of analyzed windows compared to running a sliding window algorithm (even convolutional), however it is still rather slow and there has been much work to create faster region proposal algorithms:

* The [R-CNN](https://arxiv.org/abs/1311.2524) propose regions and classify each region on at a time. It produce the output label and a bounding box. In fact R-CNN doesn't trust the bounding box that it is provided with and instead tries to define its own. Its downside is that it is slow and computationally heavy
* [Fast R-CNN](https://arxiv.org/abs/1504.08083) is a faster implementation of the R-CNN algorithm that also propose regions but, contrary to the classic R-CNN method, uses a convolutional implementation of the sliding window to classify all the proposed regions. While being faster than R-CNN, the region proposal step still is slow
* [Faster R-CNN](https://arxiv.org/pdf/1506.01497) uses convolutional implmentation of the semantic segmentation. While being significantly faster than the Fast R-CNN method, it still is slower than the YOLO algorithm.

## Semantic segmentation
A fully convolutional implementation of semantic segmentation is an idea that has evolved beyond its initial purpose of feeding regions to a bounding box prediction algorithm. Some applications require the semantic segmentation output itself: for example, for a self-driving car it may be more useful to know exactly which pixels of the image correspond to a road rather than drawing a bounding box around the road (<a href="#fig:semseg">Figure 111</a>). Another field that benefits from semantic segmentation is medical images (e.g. X-ray, MRI) analysis, which can detect irregularities and detect disorders.

In semantic segmentation you have per-pixel class labels. Suppose you want to be able to detect the region of a picture occupied by a car, you would have just to labels: $1$ for the car and $0$ for the background. in this case the task of the segmentation algorithm would be to output either 1 or 0 for every pixel (<a href="#fig:segmclasses">Figure 112</a>).


    

<figure id="fig:segmclasses">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_5_0.svg" alt="png">
    <figcaption>Figure 112. An input picture with overlayed a sample of the per-pixel classes outputted by the pspnet segmentation algorithm (A). The full segmentation output color-coded with the car (class 7 in this model) i purple and the background (class 0) in yellow.</figcaption>
</figure>

So in a segmentation algorithm the output that we want to train the network to produce is usually a large matrix, in contrast with the relatively small output dimensions of the YOLO algorithm. In all convolutional architecture seen until now, the depth of the volume gradually increase while going deeper in the network layers and the final volume of the network usually as very small width and height. The task of a semantic segmentation network is to output very detailed localization information, so the width height of its final layer needs to have large height and width (a large resolution). To achieve this, the architecture of a semantic segmentation network has its first layers similar to a classic CONV networks, where the number of channels grow while the width and height shrink; however the volumes are then gradually upscaled and the width and height are restored to large values. To achieve this, semantic segmentation networks employ a method called **transpose convolutions**


    

<figure id="fig:unetarch">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_7_0.svg" alt="png">
    <figcaption>Figure 113. Simplified representation of variation of volume dimensions along the layers of a semantic segmentation network. Early layers behave like a normal CONV net with the volume growing in depth while shrinking in width and height. Final layers network need to scale up the width and height of the volume to reach a sufficient resolution for the segmentation output.</figcaption>
</figure>

## Transpose convolutions



```python

```
