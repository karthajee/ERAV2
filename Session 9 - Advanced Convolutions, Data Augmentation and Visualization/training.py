from tqdm import tqdm
import torch
import torch.nn.functional as F

class Trainer:
    """
    A custom training utility class for PyTorch models. It facilitates the training and testing processes
    for a given model, handling the batch-wise forward and backward passes, loss calculations, and accuracy
    tracking over epochs.

    Attributes:
    - train_losses (list): A list that tracks the loss of each batch during training.
    - test_losses (list): A list that tracks the average loss per epoch during testing.
    - train_acc (list): A list that tracks the training accuracy after each batch.
    - test_acc (list): A list that tracks the testing accuracy after each epoch.

    Methods:
    - train(model, device, optimizer, train_loader): Conducts the training process for a single epoch.
    - test(model, device, test_loader): Evaluates the model on a given dataset.
    """

    def __init__(self):
        """
        Initializes the Trainer instance, setting up containers for tracking training and testing metrics.
        """
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def train(self, model, device, optimizer, train_loader):
        """
        Executes one epoch of training: iterates over the training dataset, performs forward and backward
        passes, and updates model parameters.

        Parameters:
        - model (torch.nn.Module): The neural network model to train.
        - device (torch.device): The device (CPU, GPU, etc.) on which the model and data are located.
        - optimizer (torch.optim.Optimizer): The optimizer used for parameter updates.
        - train_loader (torch.utils.data.DataLoader): The DataLoader providing training data batches.
        """
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = F.nll_loss(y_pred, target)
            self.train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)

    def test(self, model, device, test_loader):
        """
        Evaluates the model's performance on the test dataset for one epoch, reporting loss and accuracy.

        Parameters:
        - model (torch.nn.Module): The neural network model to evaluate.
        - device (torch.device): The device (CPU, GPU, etc.) on which the model and data are located.
        - test_loader (torch.utils.data.DataLoader): The DataLoader providing test data batches.
        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        self.test_acc.append(100. * correct / len(test_loader.dataset))