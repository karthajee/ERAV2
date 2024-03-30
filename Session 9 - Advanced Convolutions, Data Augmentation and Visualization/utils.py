import torch
import platform
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np

import platform
import torch
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt

def get_device():
    """
    Determines the most appropriate device for torch computations based on the available hardware.
    
    This function checks the system's platform and available hardware accelerators (GPU/MPS),
    preferring GPU on non-Mac systems and MPS (Apple's Metal Performance Shaders) on Mac systems 
    when available. If neither is available, it defaults to CPU.

    Returns:
    - device (torch.device): The torch device object indicating the selected hardware device.
    """
    if platform.system().lower() == 'darwin':
        use_gpu = torch.backends.mps.is_built()
        dev_name = "mps"
    elif torch.cuda.is_available():    
        dev_name = "cuda"
    else:
        dev_name = "cpu"
    device = torch.device(dev_name)
    return device

def print_summary(model, in_size=(1, 3, 32, 32)):
    """
    Prints a summary of the model including the structure and the number of trainable parameters.

    This function leverages the `torchinfo.summary` function to provide a detailed overview
    of the model's layers, including their output shapes and parameter counts, given an input size.

    Parameters:
    - model (torch.nn.Module): The model to summarize.
    - in_size (tuple, optional): The size of the input tensor (including batch size), default is (1, 3, 32, 32).
    """
    print(summary(model, in_size))

def visualize(Trainer):
    """
    Visualizes training and testing losses and accuracies from a training instance.

    Given a Trainer instance (assumed to contain training/testing losses and accuracies),
    this function plots five charts: training loss, testing loss, training accuracy, testing accuracy, 
    and the absolute difference between training and testing accuracies (normalized by 100) across epochs.

    Parameters:
    - Trainer (object): An object expected to have `train_losses`, `test_losses`, `train_acc`, and `test_acc`
      lists populated during the training process.
    """
    train_test_diff = [np.abs(tr-te)/100 for tr, te in zip(Trainer.train_acc, Trainer.test_acc)]
    fig, axs = plt.subplots(ncols=5, figsize=(25,5))
    axs[0].plot(Trainer.train_losses)
    axs[0].set(title="Train loss", xlabel="Steps")

    axs[1].plot(Trainer.test_losses)
    axs[1].set(title="Test loss", xlabel="Epochs")

    axs[2].plot(Trainer.train_acc)
    axs[2].set(title="Train accuracy", xlabel="Steps")

    axs[3].plot(Trainer.test_acc)
    axs[3].set(title="Test accuracy", xlabel="Epochs")

    axs[4].plot(train_test_diff)
    axs[4].set(title="Train-test acc difference", xlabel="Epochs")

    plt.show()
