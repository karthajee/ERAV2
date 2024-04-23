import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class Net(nn.Module):
    """
    A convolutional neural network model extending the PyTorch nn.Module class, 
    designed for image classification tasks. This network consists of four convolutional 
    layers followed by two fully connected layers.

    The architecture is structured to progressively reduce the spatial dimensions 
    of the input image through convolutions and pooling, while increasing the depth 
    (number of channels) to learn more complex features. The final fully connected layers 
    aim to perform classification based on the features extracted by the convolutional layers.

    Attributes
    ----------
    conv1 : nn.Conv2d
        The first convolutional layer with 32 filters, kernel size 3, and stride 1.
    conv2 : nn.Conv2d
        The second convolutional layer with 64 filters, kernel size 3, and stride 1.
    conv3 : nn.Conv2d
        The third convolutional layer with 128 filters, kernel size 3, and stride 2.
    conv4 : nn.Conv2d
        The fourth convolutional layer with 256 filters, kernel size 3, and stride 1.
    fc1 : nn.Linear
        The first fully connected layer with an input size of 4096 (flattened from the output of the last pooling layer)
        and an output size of 50.
    fc2 : nn.Linear
        The second fully connected layer with an input size of 50 and an output size of 10, typically corresponding
        to the number of classes for classification.

    Methods
    -------
    forward(x)
        Defines the forward pass of the network. It accepts a tensor `x` as input, processes it through
        the convolutional and fully connected layers, and returns the log softmax of the output, which is
        useful for classification tasks.
    """

    def __init__(self):
        """
        Initializes the network architecture by defining the convolutional and fully connected layers.
        It utilizes the super() function to call the __init__() method of the nn.Module class, allowing
        the use of its functionalities.
        """
        super(Net, self).__init__()
        # r_in:1, n_in:28, j_in:1, s:1, r_out:3, n_out: 26, j_out:1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # r_in:3, n_in:26, j_in:1, s:1, r_out:5, n_out: 24, j_out:1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        ## MaxPool2D (r_in:5, n_in:24, j_in:1, s:2, r_out:7, n_out: 11.5 >> 12,  j_out: 2)
        # r_in:7, n_in: 12, j_in: 2, r_out:11, n_out: 10, j_out: 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # r_in:11, n_in: 10, j_in: 2, s:1, r_out:15, n_out: 8, j_out:2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        ## MaxPool2D (r_in:15, n_in: 8, j_in:2, s:2, r_out: 19, n_out: 3.5 >> 4, j_out: 4)
        self.fc1 = nn.Linear(4096, 50)  # 256*4*4 = 4096, considering the spatial size of the output from conv4 after pooling
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Implements the forward pass of the network. The input x passes through each layer sequentially.
        The convolutional layers are followed by ReLU activation functions, and the second and fourth convolutional
        layers are followed by max pooling operations. The output of the last convolutional layer is flattened
        before being passed through two fully connected layers with a ReLU activation function after the first.
        The final output is passed through a log softmax function.

        Parameters
        ----------
        x : Torch Tensor
            The input tensor containing the batch of images to be processed. The tensor is expected to have
            a shape [batch_size, channels, height, width].

        Returns
        -------
        Torch Tensor
            The log softmax of the network output across the classes. This output is typically used in conjunction
            with a loss function like negative log likelihood for classification tasks.
        """
        x = F.relu(self.conv1(x))  # Activation after conv1
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Conv2 followed by pooling
        x = F.relu(self.conv3(x))  # Activation after conv3
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # Conv4 followed by pooling
        x = x.view(-1, 4096)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))  # Activation after fc1
        x = self.fc2(x)  # Output from fc2
        return F.log_softmax(x, dim=1)

def GetCorrectPredCount(pPrediction, pLabels):
    """
    Calculates the number of correct predictions by comparing the predicted labels
    against the ground truth labels. The function is primarily used in the context of evaluating
    classification models, where it determines how many predictions match the actual labels.

    Parameters
    ----------
    pPrediction : torch.Tensor
        A 2D tensor containing the log softmax outputs of a neural network for a batch of inputs.
        Each row corresponds to a single input's prediction across all classes (i.e., the likelihood
        of each class). The shape of the tensor is typically [batch_size, number_of_classes].
    pLabels : torch.Tensor
        A 1D tensor containing the ground truth labels for a batch of inputs. Each element in the tensor
        is the label index of the true class for the corresponding input. The shape of the tensor is
        typically [batch_size], where each value is an integer representing the class index.

    Returns
    -------
    int
        The total number of correct predictions in the batch. This is determined by comparing the index
        of the maximum log softmax value in each row of `pPrediction` (which represents the predicted class)
        against the actual class index in `pLabels`. The function returns this count as an integer, which
        can be used to calculate metrics like accuracy. 
    """
    # Identifies the predicted class index for each input by finding the maximum log softmax value's index.
    predicted_labels = pPrediction.argmax(dim=1)
    # Compares the predicted class indices with the actual class indices.
    correct_predictions = predicted_labels.eq(pLabels)
    # Sums up the correct predictions to get the total count.
    total_correct = correct_predictions.sum().item()
    
    return total_correct

def train(model, device, train_loader, optimizer, train_acc, train_losses):
    """
    Trains a neural network model on a dataset for one epoch. This function processes the input data in batches,
    computes the loss, performs backpropagation, and updates the model parameters. It also calculates and stores
    the training accuracy and loss for each epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model that will be trained.
    ?? device : str
    ??    The device to which tensors will be moved for model training. Typically 'cuda' or 'cpu'.
    train_loader : torch.utils.data.DataLoader
        The DataLoader that provides batches of the training data.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters based on the computed gradients.
    train_acc : list
        A list that stores the training accuracy values after each epoch. This function appends the accuracy
        for the current epoch to this list.
    train_losses : list
        A list that stores the training loss values after each epoch. This function appends the average loss
        for the current epoch to this list.

    Notes
    -----
    This function internally uses a progress bar (via tqdm) to provide visual feedback on the training progress
    within an epoch. The progress bar displays the loss and accuracy after processing each batch.

    The training process includes the following steps for each batch:
    1. Data and target labels are moved to the specified device (CPU or GPU).
    2. The optimizer's gradient buffers are reset.
    3. A forward pass is performed to compute the predictions.
    4. The loss is calculated using the predictions and the true labels.
    5. Backpropagation is performed to compute the gradients.
    6. The optimizer updates the model parameters based on the gradients.
    7. The number of correct predictions and the total loss are updated.
    8. Training accuracy and loss statistics are updated and displayed in the progress bar.

    The average loss and accuracy for the epoch are appended to the `train_losses` and `train_acc` lists,
    respectively, for later analysis or visualization.
    """
    
    model.train()  # Set the model to training mode
    pbar = tqdm(train_loader)  # Initialize progress bar

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)  # Move data & labels to device
        
        optimizer.zero_grad()  # Clear existing gradients
        pred = model(data)  # Compute the model output
        
        loss = F.nll_loss(pred, target)  # Calculate loss
        train_loss += loss.item()  # Update total loss
        loss.backward()  # Perform backpropagation
        optimizer.step()  # Update model parameters
        
        correct += GetCorrectPredCount(pred, target)  # Update correct predictions count
        processed += len(data)  # Update processed samples count
        
        # Update progress bar with current loss and accuracy
        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
    train_acc.append(100. * correct / processed)  # Append epoch accuracy
    train_losses.append(train_loss / len(train_loader))  # Append average epoch loss

def test(model, device, test_loader, test_acc, test_losses):
    """
    Evaluates the performance of a trained neural network model on a test dataset. This function
    computes the loss and accuracy of the model predictions against the true labels of the test dataset.
    It operates in evaluation mode, meaning it doesn't compute gradients which makes it more memory-efficient
    and faster for the testing phase.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be evaluated.
    ?? device : str
    ??    The device ('cuda' or 'cpu') on which to perform the evaluation, determining where the data and model are loaded.
    test_loader : torch.utils.data.DataLoader
        The DataLoader providing batches of the test data.
    test_acc : list
        A list to which the accuracy of the model on the test dataset will be appended. Accuracy is calculated as
        the percentage of correctly predicted labels over the total number of test dataset examples.
    test_losses : list
        A list to which the average loss of the model on the test dataset will be appended. The loss is averaged
        over all examples in the test dataset.

    For each batch in the `test_loader`, it moves the data to the specified `device`, computes the model's
    predictions, calculates the batch's loss, and counts the number of correct predictions. The total loss and
    total number of correct predictions are used to compute the average loss and accuracy across the entire
    test dataset. These metrics are appended to `test_losses` and `test_acc` lists respectively, and printed
    out for the user.

    """
    
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to the specified device
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Calculate and sum up batch loss
            correct += GetCorrectPredCount(output, target)  # Count correct predictions

    test_loss /= len(test_loader.dataset)  # Calculate average loss

    test_acc.append(100. * correct / len(test_loader.dataset))  # Append accuracy to list
    test_losses.append(test_loss)  # Append average loss to list

    # Print average loss and accuracy
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)')
     
def train_test_run(model, device, train_loader, test_loader, num_epochs,
                   train_acc, train_losses, test_acc, test_losses):
    """
    Conducts a training and testing cycle on the given model for a specified number of epochs. Each epoch consists
    of a full training pass over the entire training dataset followed by a testing pass over the entire test dataset.
    This function also configures an optimizer and a learning rate scheduler to optimize the model parameters and
    adjust the learning rate throughout training, respectively.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained and evaluated.
    device : str
        The device ('cuda' or 'cpu') on which the model training and evaluation will be performed.
    train_loader : torch.utils.data.DataLoader
        DataLoader for providing batches of training data.
    test_loader : torch.utils.data.DataLoader
        DataLoader for providing batches of testing data.
    num_epochs : int
        The number of complete passes through the entire training dataset.
    train_acc : list
        A list to store the training accuracy after each epoch.
    train_losses : list
        A list to store the training loss after each epoch.
    test_acc : list
        A list to store the testing accuracy after each epoch.
    test_losses : list
        A list to store the testing loss after each epoch.

    Notes
    -----
    - The optimizer used here is SGD (Stochastic Gradient Descent) with a specified learning rate and momentum. 
      These hyperparameters are crucial for the convergence and performance of the training process.
    - A learning rate scheduler is used to decrease the learning rate by a factor of 0.1 every 15 epochs. Adjusting
      the learning rate can help the model to converge faster and possibly achieve better performance by taking
      smaller steps in the optimization landscape as training progresses.
    - The training and testing loops are abstracted into separate functions, `train` and `test`, which are called 
      for each epoch. This separation makes the code modular and easier to manage.
    - After each epoch, the scheduler's `step` method is called to potentially decrease the learning rate according 
      to the predefined schedule.

    """
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Define the optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)  # Define the LR scheduler

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        
        train(model, device, train_loader, optimizer, train_acc, train_losses)  # Training phase
        test(model, device, test_loader, test_acc, test_losses)  # Testing phase
        
        scheduler.step()  # Adjust the learning rate
