from dataset import BratsDataset                        # To instantiate the BratsDataset
from torch.utils.data import DataLoader                 # To create a DataLoader for the neural network
from checkTrainerLoader import check_train_loader       # For debugging the train loader
import random
import torch

CHECK_TRAIN_LOADER = False  # Flag to enable/disable the train loader check

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            axis = random.choice([0, 1, 2])  # Scegli l'asse per il flip
            image = torch.flip(image, [axis])
            mask = torch.flip(mask, [axis])
        return image, mask

class RandomNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        noise = torch.randn_like(image) * self.std + self.mean
        image = image + noise
        return image, mask

class RandomRotation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # Numero di rotazioni di 90 gradi
            axis = random.choice([0, 1, 2])  # Scelta dell'asse di rotazione (Z, Y o X)
            if axis == 0:
                image = torch.rot90(image, k, [1, 2])  # Rotazione lungo asse Z
                mask = torch.rot90(mask, k, [1, 2])
            elif axis == 1:
                image = torch.rot90(image, k, [0, 2])  # Rotazione lungo asse Y
                mask = torch.rot90(mask, k, [0, 2])
            elif axis == 2:
                image = torch.rot90(image, k, [0, 1])  # Rotazione lungo asse X
                mask = torch.rot90(mask, k, [0, 1])
        return image, mask


class Compose:
    """Applica una serie di trasformazioni in sequenza."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


def step_two(test = False):
    # Construct tensor Datasets for the training set and validation set
    # Obtain DataLoaders from the two Datasets

    augment = 1
    
    transform = Compose([
        RandomFlip(p=0.5),            # Flip casuale con probabilità 50%
        RandomRotation(p=0.5),        # Rotazione casuale con probabilità 50%
        RandomNoise(mean=0, std=0.1)  # Rumore gaussiano
    ])

    batch_size = int(input("Input batch size: "))
    num_workers = int(input("Input num_workers: "))

    # Removed drop_last = True to try using all available volumes
    train_set = BratsDataset('input_data_128_split/train', transform if augment else None) # Instantiate training dataset
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)  # Create training DataLoader
    val_set = BratsDataset('input_data_128_split/val', transform if not test and augment else None) # Instantiate validation dataset
    val_loader = DataLoader(val_set, batch_size, shuffle=test, num_workers=num_workers, pin_memory=True)  # Create validation DataLoader

    if CHECK_TRAIN_LOADER:  # Check if train loader check is enabled
        check_train_loader(train_loader)  # Debug the training DataLoader

    return train_loader, val_loader  # Return the training and validation loaders

if __name__ == '__main__':
    step_two()
