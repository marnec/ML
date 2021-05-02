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

In an R-CNN a segmentation filter is first applied to the input image to detect different areas on the image (panel B of <a href="#fig:semseg">Figure 111</a>). This segmentation step produces a number of blobs (usually some thousands). Bounding box are draw around each of these blobs and an object detection algorithm tries to detect objects inside the bounding box.




    <matplotlib.patches.Rectangle at 0x7f945147ee10>




    

<figure id="fig:semseg">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_2_1.svg" alt="png">
    <figcaption>Figure 111. An example of a very precise semantic segmentation (B) achieved using 3D-data from a Lidar and a video feed from a moving car (A). In a classic sliding window approach (convolutional or not) most of the windows will contain uninteresting data (red bounding box), while only some areas will actually contain an object (blue bounding box)</figcaption>
</figure>

This method hugely reduces the number of analyzed windows compared to running a sliding window algorithm (even convolutional), however it is still rather slow and there has been much work to create faster region proposal algorithms:

* The [R-CNN](https://arxiv.org/abs/1311.2524) propose regions and classify each region on at a time. It produce the output label and a bounding box. In fact R-CNN doesn't trust the bounding box that it is provided with and instead tries to define its own. Its downside is that it is slow and computationally heavy
* [Fast R-CNN](https://arxiv.org/abs/1504.08083) is a faster implementation of the R-CNN algorithm that also propose regions but, contrary to the classic R-CNN method, uses a convolutional implementation of the sliding window to classify all the proposed regions. While being faster than R-CNN, the region proposal step still is slow
* [Faster R-CNN](https://arxiv.org/pdf/1506.01497) uses convolutional implmentation of the semantic segmentation. While being significantly faster than the Fast R-CNN method, it still is slower than the YOLO algorithm.

## Semantic segmentation
A fully convolutional implementation of semantic segmentation is an idea that has evolved beyond its initial purpose of feeding regions to a bounding box prediction algorithm. Some applications require the semantic segmentation output itself: for example, for a self-driving car it may be more useful to know exactly which pixels of the image correspond to a road rather than drawing a bounding box around the road.
