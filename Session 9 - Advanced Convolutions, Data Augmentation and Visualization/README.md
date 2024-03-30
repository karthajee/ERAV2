# Session 9: Advanced Convolutions, Data Augmentation and Visualization

## Assignment Overview

The goal is to build a neural network that not only obtains a 85% accuracy on the CIFAR10 dataset but also satisfies the following constraints:
- Number of parameters < 200K
- C1C2C3C4O architecture where $C_i$ refers to a convolution block made up of 3 convolution layers and $O$ should necessarily contain GAP
- Contains at least one dilated convolution and depthwise separable convolution layer
- Total RF > 44
- 3 albumentation transforms - Horizontal Flip, Shift Scale Rotate and Coarse Dropout (~Cutout)
- Modular code

## Solution Repo Overview

| File | Description |
| --- | --- |
| `s9_main.ipynb` | Experiment notebook that demonstrates the solution |
| `model.py` | Code for custom network class and layer components |
| `training.py` | Contains Trainer class with corresponding train and test methods |
| `transform.py` | Contains Transform class for the required albumentation transforms  |
| `utils.py` | Convenience functions for device management, printing etc. |
| `./weights` | Directory that contains 2 saved model checkpoints | 

## Model Summary

````
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
Net                                                [1, 10]                   --
├─ConvBlock: 1-1                                   [1, 32, 16, 16]           --
│    └─Sequential: 2-1                             [1, 32, 32, 32]           --
│    │    └─DepthwiseSeparableConv2d: 3-1          [1, 32, 32, 32]           123
│    │    └─ReLU: 3-2                              [1, 32, 32, 32]           --
│    │    └─BatchNorm2d: 3-3                       [1, 32, 32, 32]           64
│    │    └─Dropout: 3-4                           [1, 32, 32, 32]           --
│    └─Sequential: 2-2                             [1, 32, 32, 32]           --
│    │    └─DepthwiseSeparableConv2d: 3-5          [1, 32, 32, 32]           1,312
│    │    └─ReLU: 3-6                              [1, 32, 32, 32]           --
│    │    └─BatchNorm2d: 3-7                       [1, 32, 32, 32]           64
│    │    └─Dropout: 3-8                           [1, 32, 32, 32]           --
│    └─Sequential: 2-3                             [1, 32, 16, 16]           --
│    │    └─DepthwiseSeparableConv2d: 3-9          [1, 32, 16, 16]           1,312
│    │    └─ReLU: 3-10                             [1, 32, 16, 16]           --
│    │    └─BatchNorm2d: 3-11                      [1, 32, 16, 16]           64
│    │    └─Dropout: 3-12                          [1, 32, 16, 16]           --
├─ConvBlock: 1-2                                   [1, 64, 8, 8]             --
│    └─Sequential: 2-4                             [1, 64, 16, 16]           --
│    │    └─DepthwiseSeparableConv2d: 3-13         [1, 64, 16, 16]           2,336
│    │    └─ReLU: 3-14                             [1, 64, 16, 16]           --
│    │    └─BatchNorm2d: 3-15                      [1, 64, 16, 16]           128
│    │    └─Dropout: 3-16                          [1, 64, 16, 16]           --
│    └─Sequential: 2-5                             [1, 64, 16, 16]           --
│    │    └─DepthwiseSeparableConv2d: 3-17         [1, 64, 16, 16]           4,672
│    │    └─ReLU: 3-18                             [1, 64, 16, 16]           --
│    │    └─BatchNorm2d: 3-19                      [1, 64, 16, 16]           128
│    │    └─Dropout: 3-20                          [1, 64, 16, 16]           --
│    └─Sequential: 2-6                             [1, 64, 8, 8]             --
│    │    └─DepthwiseSeparableConv2d: 3-21         [1, 64, 8, 8]             4,672
│    │    └─ReLU: 3-22                             [1, 64, 8, 8]             --
│    │    └─BatchNorm2d: 3-23                      [1, 64, 8, 8]             128
│    │    └─Dropout: 3-24                          [1, 64, 8, 8]             --
├─ConvBlock: 1-3                                   [1, 128, 4, 4]            --
│    └─Sequential: 2-7                             [1, 128, 8, 8]            --
│    │    └─DepthwiseSeparableConv2d: 3-25         [1, 128, 8, 8]            8,768
│    │    └─ReLU: 3-26                             [1, 128, 8, 8]            --
│    │    └─BatchNorm2d: 3-27                      [1, 128, 8, 8]            256
│    │    └─Dropout: 3-28                          [1, 128, 8, 8]            --
│    └─Sequential: 2-8                             [1, 128, 8, 8]            --
│    │    └─DepthwiseSeparableConv2d: 3-29         [1, 128, 8, 8]            17,536
│    │    └─ReLU: 3-30                             [1, 128, 8, 8]            --
│    │    └─BatchNorm2d: 3-31                      [1, 128, 8, 8]            256
│    │    └─Dropout: 3-32                          [1, 128, 8, 8]            --
│    └─Sequential: 2-9                             [1, 128, 4, 4]            --
│    │    └─DepthwiseSeparableConv2d: 3-33         [1, 128, 4, 4]            17,536
│    │    └─ReLU: 3-34                             [1, 128, 4, 4]            --
│    │    └─BatchNorm2d: 3-35                      [1, 128, 4, 4]            256
│    │    └─Dropout: 3-36                          [1, 128, 4, 4]            --
├─ConvBlock: 1-4                                   [1, 224, 2, 2]            --
│    └─Sequential: 2-10                            [1, 224, 4, 4]            --
│    │    └─DepthwiseSeparableConv2d: 3-37         [1, 224, 4, 4]            29,824
│    │    └─ReLU: 3-38                             [1, 224, 4, 4]            --
│    │    └─BatchNorm2d: 3-39                      [1, 224, 4, 4]            448
│    │    └─Dropout: 3-40                          [1, 224, 4, 4]            --
│    └─Sequential: 2-11                            [1, 224, 4, 4]            --
│    │    └─DepthwiseSeparableConv2d: 3-41         [1, 224, 4, 4]            52,192
│    │    └─ReLU: 3-42                             [1, 224, 4, 4]            --
│    │    └─BatchNorm2d: 3-43                      [1, 224, 4, 4]            448
│    │    └─Dropout: 3-44                          [1, 224, 4, 4]            --
│    └─Sequential: 2-12                            [1, 224, 2, 2]            --
│    │    └─DepthwiseSeparableConv2d: 3-45         [1, 224, 2, 2]            52,192
│    │    └─ReLU: 3-46                             [1, 224, 2, 2]            --
│    │    └─BatchNorm2d: 3-47                      [1, 224, 2, 2]            448
│    │    └─Dropout: 3-48                          [1, 224, 2, 2]            --
├─Sequential: 1-5                                  [1, 10, 1, 1]             --
│    └─AdaptiveAvgPool2d: 2-13                     [1, 224, 1, 1]            --
│    └─Conv2d: 2-14                                [1, 10, 1, 1]             2,250
====================================================================================================
Total params: 197,413
Trainable params: 197,413
Non-trainable params: 0
Total mult-adds (M): 7.39
====================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 2.94
Params size (MB): 0.79
Estimated Total Size (MB): 3.74
````

## Design Choices

- Used depthwise separable convolution layers as the default convolution layer in each block to maximize parameters given max parameter constraint
- Implemented dilated convolution layer as the 3rd layer in each conv block
- Albumentation transformations made interoperable with torchvision dataloaders by creating a custom class around `A.Compose(...)
- Stopped and restarted training with fresh optimizer and OneCycleLR scheduler (with a lower max learning rate than the first training period) helped jumpstart the model from a local minima it was stuck in

## Training Curves

![Train Test Loss Curves](img/output.png)

## Misclassified Images