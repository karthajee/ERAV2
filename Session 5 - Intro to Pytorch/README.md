# Assignment Submission for Session 5 - Intro to Pytorch

Goal of the assignment is to refactor the [Session 4 jupyter notebook](https://colab.research.google.com/drive/1oVt7T6tb90Y1EXvFaIWgm72emqZZPXi4?usp=sharing) into 3 components:
- A `model.py` script containing all model-related code
- A `utils.py` script containing convenience and non-model-related code
- A `S5.ipynb` notebook for demonstration

Subsequent sections explain the different components

## `model.py`

This script contains code for initializing the neural network model and its train & test functionality. Model structure is as below below:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

This model can work with any dataset containing (28, 28) images and 10 class labels. Certain changes were made to the Session 4 code to make this separation executable. For e.g., the function definition of `train(...)` was altered to accept lists to store accuracy and loss values. It can be argued that train and test functionality should be included in utility because it is "model-agnostic" - I fall on the other side of the argument as they are fundamental to actual model usage

## `utils.py`

The assignment requires us to work on the MNIST dataset. This script contains code for downloading the dataset, transforming them, creating dataloaders etc. Additionally, the functionality to plot train & test performance metrics have also been moved here as plotting is typically seen as a utility function

## `S5.ipynb`

The notebook imports the necessary libraries and can be executed from start to finish to produce equivalent results to Session 4 assignment

## Parting Thoughts

As Rohan mentioned in the class, refactoring leads to massive efficiency and efficacy gains. This is even more salient during the initial stages of the project itself. Otherwise:

![](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjgzYTAyNmUwYmNhNWYyYWUwMWFhYmMyMGEzNTg0Yjc2MzNlZDUyZiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/l0HlGmv4WqldO9c5y/giphy.gif)
