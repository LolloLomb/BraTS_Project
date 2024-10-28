import glob                                             # To save paths in subdirectories
import splitfolders                                     # To split paths between train and validation
from new2024.cropAndSave import crop_and_save                 # To crop images and save them
from sklearn.preprocessing import MinMaxScaler          # To impose which class we want to accept as input for crop_and_save

def step_one():

    dir = 'new2024/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    scaler = MinMaxScaler

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

if __name__ == '__main__':
    step_one()