---
layout: default
title: "Deep Learning - CNN - U-Net"
categories: deeplearning
permalink: /ML39/
order: 39
comments: true
---

# U-Net
The U-net architecture is one of the more important and foundational neural network for object detection today

## Region proposal
In the body of work about object detection an influential idea that has been proposed is **region proposal**. One of the classic region proposal algorithm is called R-CNN.

In a classic sliding window algorithm we would roll a sliding window across the whole input image, starting from the top-left corner all the way to the bottom-right corner. While we have seen that a convolutional implementation of the sliding window exists, it remains the problem that most of the windows analyzed will contain uninteresting data. 

In an R-CNN an **unsupervised segmentation** filter is first applied to the input image to detect different areas on the image (panel B of <a href="#fig:semseg">Figure 113</a>). This segmentation step produces a number of blobs (usually some thousands). Bounding box are draw around each of these blobs and an object detection algorithm tries to detect objects inside the bounding box.


    

<figure id="fig:semseg">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_2_0.svg" alt="png">
    <figcaption>Figure 113. An example of a very precise semantic segmentation (B) achieved using 3D-data from a Lidar and a video feed from a moving car (A). In a classic sliding window approach (convolutional or not) most of the windows will contain uninteresting data (red bounding box), while only some areas will actually contain an object (blue bounding box)</figcaption>
</figure>

This method hugely reduces the number of analyzed windows compared to running a sliding window algorithm (even convolutional), however it is still rather slow and there has been much work to create faster region proposal algorithms:

* The [R-CNN](https://arxiv.org/abs/1311.2524) propose regions and classify each region on at a time. It produce the output label and a bounding box. In fact R-CNN doesn't trust the bounding box that it is provided with and instead tries to define its own. Its downside is that it is slow and computationally heavy
* [Fast R-CNN](https://arxiv.org/abs/1504.08083) is a faster implementation of the R-CNN algorithm that also propose regions but, contrary to the classic R-CNN method, uses a convolutional implementation of the sliding window to classify all the proposed regions. While being faster than R-CNN, the region proposal step still is slow
* [Faster R-CNN](https://arxiv.org/pdf/1506.01497) uses convolutional implmentation of the semantic segmentation. While being significantly faster than the Fast R-CNN method, it still is slower than the YOLO algorithm.

## Semantic segmentation
A fully convolutional implementation of semantic segmentation is an idea that has evolved beyond its initial purpose of feeding regions to a bounding box prediction algorithm. Some applications require the semantic segmentation output itself: for example, for a self-driving car it may be more useful to know exactly which pixels of the image correspond to a road rather than drawing a bounding box around the road (<a href="#fig:semseg">Figure 113</a>). Another field that benefits from semantic segmentation is medical images (e.g. X-ray, MRI) analysis, which can detect irregularities and detect disorders.

In semantic segmentation you have per-pixel class labels. Suppose you want to be able to detect the region of a picture occupied by a car, you would have just to labels: $1$ for the car and $0$ for the background. in this case the task of the segmentation algorithm would be to output either 1 or 0 for every pixel (<a href="#fig:segmclasses">Figure 114</a>).


    

<figure id="fig:segmclasses">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_5_0.svg" alt="png">
    <figcaption>Figure 114. An input picture with overlayed a sample of the per-pixel classes outputted by the pspnet segmentation algorithm (A). The full segmentation output color-coded with the car (class 7 in this model) i purple and the background (class 0) in yellow.</figcaption>
</figure>

So in a segmentation algorithm the output that we want to train the network to produce is usually a large matrix, in contrast with the relatively small output dimensions of the YOLO algorithm. In all convolutional architecture seen until now, the depth of the volume gradually increase while going deeper in the network layers and the final volume of the network usually as very small width and height. The task of a semantic segmentation network is to output very detailed localization information, so the width height of its final layer needs to have large height and width (a large resolution). To achieve this, the architecture of a semantic segmentation network has its first layers similar to a classic CONV networks, where the number of channels grow while the width and height shrink; however the volumes are then gradually upscaled and the width and height are restored to large values. To achieve this, semantic segmentation networks employ a method called **transpose convolutions**


    

<figure id="fig:unetarch">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_7_0.svg" alt="png">
    <figcaption>Figure 115. Simplified representation of variation of volume dimensions along the layers of a semantic segmentation network. Early layers behave like a normal CONV net with the volume growing in depth while shrinking in width and height. Final layers network need to scale up the width and height of the volume to reach a sufficient resolution for the segmentation output.</figcaption>
</figure>

## Transpose convolutions
The transpose convolution is a key part of semantic segmentation architectures that is used to upscale a volume to a larger width and height. For example taking the $2 \times 2$ input in panel B of <a href="#fig:transposeconv">Figure 116</a>, we can achieve a $4 \times 4$ output by applying a $3 \times 3$ filter with padding $p=1$ and stride $s=2$. Transpose convolution achieve this by mechanically applying the filter on the output instead than on the input as in classic convolution.


    

<figure id="fig:transposeconv">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_9_0.svg" alt="png">
    <figcaption>Figure 116. The variation of dimensionality in a normal convolution that (with valid padding) reduces the width and height of the input representation (A). The variation of dimensionality of transpose convolution increases (or upscales) the width and height of the input representation (B)</figcaption>
</figure>

A single cell of the input is multiplied with the filter and this processed filter is applied to the output representation. Looking at the example in panel B of <a href="#fig:transposeconv">Figure 116</a>, we can notice that with a stride $s=2$ we will have some overlapping cells at each convolutions; values resulting from overlapping portions of the filter in different steps on the transpose convolution are summed together in the final output.

Transpose convolution is not the only method to upscale a matrix, however it turns out to be the best performing in the context of semantic segmentation and especially for **U-nets**.

## U-net
In <a href="#fig:unetarch">Figure 115</a>, we have seen the general architecture of a U-net, with the input volume being downscaled and then upscaled again to give an output with one channel and large values of width and height.

One modification to the architecture in <a href="#fig:unetarch">Figure 115</a>. This way, activations from early layers (layer 1 in the figure) are directly copied to late layers (layer 3 in the figure). 


    

<figure id="fig:unetarchcomplete">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_12_0.svg" alt="png">
    <figcaption>Figure 117. Typical U-net architecture</figcaption>
</figure>

The reason for U-net benefiting from skip-connections is that, for next-to-final layers to decide which regions of the representation is the object to detect, two pieces of information are useful:

* high-level spatial/contextual information provided by the immediately previous layer, which should have detected the object class in the approximate region where it is placed in the input image
* Fine-grain spatial information: since late layers have low resolution (low height and width) this is provided activations from early layers, obtained through skip-connections

More in detail the U-net architecture, whose shape looks like a U (<a href="fig:unetarchdetail">figure below</a>) hence the name, takes as input a large image with 3 channels. The first layers are classic CONV layer with same pooling that maintain the height and width of the input while increasing the number of channels. Then a max-pooling layer and another set of CONV layers is applied. By repeating this process we gradually shrink the width and height and increase the channels. This process is then inverted by applying transpose convolutions in place of max-pooling. The representation is scaled back up to the original dimensions of the input image. Furthermore, skip connections link early to late layers each time the transpose convolution is applied. Finally the last layer, which is fed a volume already the dimension of the input image, is a $1 \times 1$ convolution that produces the segmentation output. The dimensions of the output layer is $h \times w \times n_c$, where $w,h$ are the widht and height of the input image and $n_c$ is the number of classes on which the network is trained.


    

<figure id="fig:unetarchdetail">
    <img src="{{site.baseurl}}/pages/ML-39-DeepLearningCNN7_files/ML-39-DeepLearningCNN7_14_0.svg" alt="png">
    <figcaption>Figure 118. Detailed architecture of the U-net. Each block represents a layer of the U-net seen along the width axis ($y=$height, $x=$channels). Horizontal arrows represent convolutions with ReLU activation function, red arrows (downwards) represent max-pooling that shrink the width and height of the representation. Green arrows (upwards) represent transpose convolution that upscale the width and height of the representation. Gray horizontal arrows represent skip connections that add the early layers activations (blue) to late layers activations (cyan)</figcaption>
</figure>