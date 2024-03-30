import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

class DepthwiseSeparableConv2d(nn.Module):
    """
    Implements a depthwise separable convolution block as a subclass of nn.Module. 
    This block is useful for reducing the computational cost and the number of parameters 
    in a convolutional neural network compared to a standard convolution layer with 
    equivalent depth. It achieves this by first applying a depthwise convolution, 
    which acts separately on each input channel, and then a pointwise convolution, 
    which mixes the channels from the depthwise convolution.

    The depthwise convolution uses a separate filter for each input channel, while the 
    pointwise convolution uses a 1x1 convolution to combine the outputs from the 
    depthwise convolution into the desired number of output channels.

    Parameters:
    - in_c (int): Number of channels in the input image.
    - kernels_per_layer (int): The number of kernel filters per input channel for the depthwise convolution.
    - out_c (int): Number of channels produced by the pointwise convolution.
    - padding (int, optional): Zero-padding added to both sides of the input. Default is 1.
    - dilation (int, optional): Spacing between kernel elements. Default is 1.
    - stride (int, optional): Stride of the convolution. Default is 1.

    Attributes:
    - depthwise (nn.Conv2d): The depthwise convolution layer with parameters specified 
      by in_c, kernels_per_layer, and the convolution parameters (padding, dilation, stride).
      Uses 'reflect' padding mode for the padding.
    - pointwise (nn.Conv2d): The pointwise convolution layer that combines the channels 
      from the depthwise convolution into the specified number of output channels.
    """

    def __init__(self, in_c, kernels_per_layer, out_c, padding=1, dilation=1, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c * kernels_per_layer, 3, padding=padding,
                                   dilation=dilation, stride=stride, groups=in_c,
                                   bias=False, padding_mode='reflect')
        self.pointwise = nn.Conv2d(in_c * kernels_per_layer, out_c, 1, bias=False)
  
    def forward(self, x):
        """
        Defines the computation performed at every call of the depthwise separable convolution block.

        Parameters:
        - x (Tensor): The input tensor of shape (batch_size, in_c, H, W) where H and W are the height 
          and width of the input plane respectively.

        Returns:
        - Tensor: The output tensor after applying depthwise followed by pointwise convolutions.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvBlock(nn.Module):
    """
    Defines a convolutional block module, leveraging depthwise separable convolutions 
    for efficiency. This module consists of three convolutional sequences. Each sequence 
    includes a depthwise separable convolution (DepthwiseSeparableConv2d) followed by 
    ReLU activation, batch normalization, and dropout for regularization.

    The first two sequences maintain the input's spatial dimensions, while the third 
    sequence applies a strided convolution to downsample its input. The output of the 
    first sequence is added to the input of the second sequence (residual connection) 
    to promote effective training deep networks by alleviating the vanishing gradient 
    problem.

    Parameters:
    - in_c (int): Number of channels in the input image.
    - out_c (int): Number of output channels for the depthwise separable convolutions.
    - kpl (int, optional): Kernels per layer parameter for depthwise convolution. Defaults to 1.
    - p (float, optional): Dropout probability for regularization. Defaults to 0.1.

    Attributes:
    - conv1 (nn.Sequential): First convolutional sequence with a depthwise separable convolution, 
      ReLU activation, batch normalization, and dropout.
    - conv2 (nn.Sequential): Second convolutional sequence similar to conv1.
    - conv3 (nn.Sequential): Third convolutional sequence, applies a strided depthwise separable 
      convolution for downsampling, followed by ReLU, batch normalization, and dropout.
    """

    def __init__(self, in_c, out_c, kpl=1, p=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(in_c, kpl, out_c),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Dropout(p)
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(out_c, kpl, out_c),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Dropout(p)
        )
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv2d(out_c, kpl, out_c, padding=2, dilation=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Dropout(p)
        )

    def forward(self, x):
        """
        Forward pass of the ConvBlock. Applies three sequential operations on the input, 
        with a residual connection between the first two convolutions.

        Parameters:
        - x (Tensor): The input tensor of shape (batch_size, in_c, H, W) where H and W 
          are the height and width of the input plane respectively.

        Returns:
        - Tensor: The output tensor after processing through the convolutional block.
        """
        x1 = self.conv1(x)
        x = x1 + self.conv2(x1)
        x = self.conv3(x)
        return x

class Net(nn.Module):
    """
    Defines a neural network model that employs a sequence of convolutional blocks 
    for processing images. This network is structured to gradually increase the 
    channel depth while reducing the spatial dimensions of the input, aiming to 
    extract meaningful features at various scales. The model concludes with an 
    adaptive average pooling layer to reduce the spatial dimensions to 1x1, followed 
    by a convolutional layer that acts as a fully connected layer for classification.

    Parameters:
    - C1_C (int, optional): Number of output channels for the first convolutional block. Default is 32.
    - C2_C (int, optional): Number of output channels for the second convolutional block. Default is 64.
    - C3_C (int, optional): Number of output channels for the third convolutional block. Default is 128.
    - C4_C (int, optional): Number of output channels for the fourth convolutional block. Default is 256.
    - kpl (int, optional): Kernels per layer parameter for depthwise convolution in ConvBlocks. Defaults to 1.
    - p (float, optional): Dropout probability in ConvBlocks for regularization. Defaults to 0.1.

    Attributes:
    - C1, C2, C3, C4 (ConvBlock): Convolutional blocks with increasing channel depth and added dropout.
    - O (nn.Sequential): Output sequence that includes an adaptive average pooling layer to reduce 
      spatial dimensions, followed by a convolutional layer for classification.

    Methods:
    - forward(x): Implements the forward pass of the model.
    - save(dir): Saves the model to a specified directory with a timestamp and channel configuration in the filename.
    """

    def __init__(self, C1_C=32, C2_C=64, C3_C=128, C4_C=256, kpl=1, p=0.1):
        super(Net, self).__init__()
        self.C1_C = C1_C
        self.C2_C = C2_C
        self.C3_C = C3_C
        self.C4_C = C4_C    

        self.C1 = ConvBlock(3, self.C1_C)
        self.C2 = ConvBlock(self.C1_C, self.C2_C, kpl, p)
        self.C3 = ConvBlock(self.C2_C, self.C3_C, kpl, p)
        self.C4 = ConvBlock(self.C3_C, self.C4_C, kpl, p)
        self.O = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(self.C4_C, 10, kernel_size=1, padding=0)
        )

    def forward(self, x):
        """
        Forward pass of the Net. Sequentially processes the input through four convolutional 
        blocks and the output sequence for classification.

        Parameters:
        - x (Tensor): The input tensor of shape (batch_size, 3, H, W) where 3 represents 
          the input channels (RGB), and H and W are the height and width of the input image.

        Returns:
        - Tensor: The output tensor with log softmax applied, representing the log probabilities of classes.
        """
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.O(x)
        x = x.view(-1, 10)  # Flatten the output for the classification layer
        o = F.log_softmax(x, dim=-1)
        return o

    def save(self, dir='weights/'):
        """
        Saves the current model state with a timestamp and channel configuration in the filename.

        Parameters:
        - dir (str, optional): Directory path where the model will be saved. Defaults to 'weights/'.
        """
        os.makedirs(dir, exist_ok=True)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"{dir}{timestr}_{self.C1_C}_{self.C2_C}_{self.C3_C}_{self.C4_C}.pt"
        torch.save(self, filepath)