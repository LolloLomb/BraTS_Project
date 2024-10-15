import glob                                             # To save paths in subdirectories
import splitfolders                                     # To split paths between train and validation
from crop_and_save import crop_and_save                 # To crop images and save them
from check_train_loader import check_train_loader       # For debugging the train loader
from dataset import BratsDataset                        # To instantiate the BratsDataset
from torch.utils.data import DataLoader                 # To create a DataLoader for the neural network
from sklearn.preprocessing import MinMaxScaler          # To impose which class we want to accept as input for crop_and_save
from unet import UNet3D                                 # To create an instance of the 3D UNet model
from dice_loss import DiceLoss                          # To measure the overlap between predicted and true masks
from lightning.pytorch.callbacks import ModelCheckpoint # To save training checkpoints
from lightning import Trainer                           # To train the model
# from torch.optim.adamw import AdamW # It is used by default in my 3D UNet

CHECK_TRAIN_LOADER = False  # Flag to enable/disable the train loader check

learning_rate = 0.001       # Learning rate for the optimizer
batch_size = 1              # Batch size for training
max_epochs = 1              # Maximum number of epochs for training

def step_one(dir, scaler):
    # For each type of data: T2, T1CE, FLAIR with the respective segmented area
    t2_list = sorted(glob.glob(f'{dir}/*/*t2.nii'))  # Collect paths for T2 images
    t1ce_list = sorted(glob.glob(f'{dir}/*/*t1ce.nii'))  # Collect paths for T1CE images
    flair_list = sorted(glob.glob(f'{dir}/*/*flair.nii'))  # Collect paths for FLAIR images
    mask_list = sorted(glob.glob(f'{dir}/*/*seg.nii'))  # Collect paths for segmentation masks

    # Crop volumes after merging and normalize them using MinMaxScaler, creating a temporary train folder
    crop_and_save(t2_list, t1ce_list, flair_list, mask_list, 0.01, scaler, 'train')

    input_folder = 'new2024/input_data_128/train'  # Folder containing input data
    output_folder = 'new2024/input_data_128_split'  # Folder for split datasets

    # Split the dataset into training and validation sets, using 75% for training and 25% for validation
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None, move=True)  # default values

def step_two(batch_size):
    # Construct tensor Datasets for the training set and validation set
    # Obtain DataLoaders from the two Datasets

    # Removed drop_last = True to try using all available volumes
    train_set = BratsDataset('input_data_128_split/train')  # Instantiate training dataset
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)  # Create training DataLoader
    val_set = BratsDataset('input_data_128_split/val')  # Instantiate validation dataset
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=0)  # Create validation DataLoader

    if CHECK_TRAIN_LOADER:  # Check if train loader check is enabled
        check_train_loader(train_loader)  # Debug the training DataLoader

    return train_loader, val_loader  # Return the training and validation loaders

def step_three(loss_fx):
    return UNet3D(3, 4, loss_fx, learning_rate)  # Instantiate the 3D UNet model with input channels, output classes, loss function, and learning rate

def main():
    # dir = 'new2024/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    # Uncomment the line below to execute step one with the specified directory and a MinMaxScaler
    # step_one(dir, scaler=MinMaxScaler())
    
    train_loader, val_loader = step_two(batch_size)  # Create training and validation loaders

    loss_fx = DiceLoss(classes=4)  # Instantiate the Dice Loss function with 4 classes
    model = step_three(loss_fx)  # Create an instance of the UNet model

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Monitor the validation loss
        dirpath='checkpoints/',  # Directory to save the checkpoints
        filename='best-model',  # Filename for the best model
        save_top_k=1,  # Save only the best model
        mode='min'  # Minimize the monitored quantity
    )

    trainer = Trainer(
        max_epochs=max_epochs,  # Set maximum epochs for training
        callbacks=[checkpoint_callback],  # Include the checkpoint callback
        devices='auto',  # Automatically choose devices (CPU or GPU)
        accelerator='gpu'  # Use CPU for training
    )

    # Start the fitting process
    trainer.fit(model, train_loader, val_loader)

# Entry point of the script
if __name__ == '__main__':
    main()  # Call the main function to execute the program
