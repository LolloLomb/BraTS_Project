import os, numpy as np, torch

class BratsDataset(torch.utils.data.Dataset):
    r"""Provide the folder name, e.g., `my/folder`
    
        From the provided folder, it finds the `images/` and `masks/` directories
        and builds a dataset by converting `.npy` files into tensors."""
    

    def __init__(self, np_file_paths):
        # Get all the .npy files from the /images and /masks directories
        self.imagesPath = os.path.join(np_file_paths, "images")
        self.masksPath = os.path.join(np_file_paths, "masks")
        # Sort so that each parallel read matches the corresponding .npy volume and mask
        self.image_files = sorted(os.listdir(self.imagesPath))
        self.mask_files = sorted(os.listdir(self.masksPath))
        assert len(self.image_files) == len(self.mask_files), "The number of images and masks must be equal"

    def __getitem__(self, index):
        # Read the .npy files and convert them into tensors
        X = np.load(os.path.join(self.imagesPath, self.image_files[index]))
        X = torch.from_numpy(X).float()
        y = np.load(os.path.join(self.masksPath, self.mask_files[index]))
        y = torch.from_numpy(y).float()
        return X, y
    

    def __len__(self):
        # Return the number of volumes read
        return len(self.image_files)
