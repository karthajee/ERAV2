import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class Transforms:
    """
    A wrapper class for albumentations transformations to make them compatible with PyTorch datasets.
    
    This class ensures that the transformations defined using the albumentations library can be
    directly used in PyTorch datasets by overriding the `__call__` method to process images.

    Parameters:
    - transforms (A.Compose): An albumentations.Compose object encapsulating the desired transformations.

    Methods:
    - __call__(img): Applies the transformations to the input image.
    """

    def __init__(self, transforms: A.Compose):
        """
        Initializes the Transforms object with the specified albumentations.Compose transformations.
        """
        self.transforms = transforms

    def __call__(self, img):
        """
        Applies the predefined transformations to an input image when the object is called.

        Parameters:
        - img (PIL.Image or numpy.ndarray): The input image to transform.

        Returns:
        - numpy.ndarray: The transformed image as a numpy array.
        """
        return self.transforms(image=np.array(img))['image']

def get_transforms(means, stds):
    """
    Defines and returns the albumentations-based transformations for training and testing datasets.
    
    The function creates a set of data augmentation transformations for the training set to introduce
    variability and robustness (e.g., horizontal flipping, random shifts, scales, rotations, and dropout).
    For the test set, it standardizes the images using normalization. Both sets of transformations include
    converting the images to PyTorch tensors.

    Parameters:
    - means (list or tuple): The mean values for each channel, used for normalization.
    - stds (list or tuple): The standard deviation values for each channel, used for normalization.

    Returns:
    - tuple: A tuple containing the training and testing transformations.
    """
    train_transforms = Transforms(A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1,
                        min_height=16, min_width=16, fill_value=means, mask_fill_value=None),
        A.Normalize(mean=means, std=stds),
        ToTensorV2(),
    ]))
    test_transforms = Transforms(A.Compose([
        A.Normalize(mean=means, std=stds),
        ToTensorV2(),
    ]))
    return train_transforms, test_transforms
