from dataset import BratsDataset                        # To instantiate the BratsDataset
from torch.utils.data import DataLoader                 # To create a DataLoader for the neural network
from check_train_loader import check_train_loader       # For debugging the train loader

CHECK_TRAIN_LOADER = False  # Flag to enable/disable the train loader check

def step_two():
    # Construct tensor Datasets for the training set and validation set
    # Obtain DataLoaders from the two Datasets

    batch_size = int(input("Input batch size: "))
    num_workers = int(input("Input num_workers: "))

    # Removed drop_last = True to try using all available volumes
    train_set = BratsDataset('input_data_128_split/train')  # Instantiate training dataset
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)  # Create training DataLoader
    val_set = BratsDataset('input_data_128_split/val')  # Instantiate validation dataset
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)  # Create validation DataLoader

    if CHECK_TRAIN_LOADER:  # Check if train loader check is enabled
        check_train_loader(train_loader)  # Debug the training DataLoader

    return train_loader, val_loader  # Return the training and validation loaders

if __name__ == '__main__':
    step_two()
