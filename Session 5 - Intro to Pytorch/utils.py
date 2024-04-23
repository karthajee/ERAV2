import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def create_train_test_dataset():
  
    """
    Function that downloads train & test splits of MNIST and returns them as Datasets
    """

    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1), # Apply center cropping 10% of the times
        transforms.Resize((28, 28)), # Resize image (esp relevant for center cropped images)
        transforms.RandomRotation((-15., 15.), fill=0), # Rotate image between -15/15 degrees, filling blank values with 0
        transforms.ToTensor(), # Convert to tensor with values scaled between 0 and 1
        transforms.Normalize((0.1307,), (0.3081,)), # Normalize with mean & std of pixel values for full data
        ])

    # Test data transformations
    # Center cropping (and hence resizing) + rotation not relevant for test data
    # as augmentation useful chiefly for creating different variations of training data
    test_transforms = transforms.Compose([
        transforms.ToTensor(), # Convert to tensor with values scaled between 0 and 1
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize with mean & std of pixel values for full data
        ])

    # Download the datasets and apply relevant transforms
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    return train_data, test_data

def create_train_test_loader(train_data, test_data, **kwargs):

    """
    Function that wraps data loaders around train & test data
    :param train_data: Training Dataset
    :param test_data: Test Dataset
    """
    
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    return train_loader, test_loader

def plot_loss_acc(train_losses, train_acc, test_losses, test_acc):

    """
    Function that plots line curves of train & test loss & accuracy curves in a 2x2 grid
    :param train_losses: List of training loss values
    :param train_acc: List of training accuracy values
    :param test_losses: List of test loss values
    :param test_acc: List of test accuracy values
    """

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")