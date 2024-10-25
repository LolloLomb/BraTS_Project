import os, numpy as np, torch

class BratsDataset(torch.utils.data.Dataset):
    r"""Provide the folder name, e.g., `my/folder`
    
        From the provided folder, it finds the `images/` and `masks/` directories
        and builds a dataset by converting `.npy` files into tensors."""
    

    def __init__(self, np_file_paths, transform = None):
        # Get all the .npy files from the /images and /masks directories
        self.imagesPath = os.path.join(np_file_paths, "images")
        self.masksPath = os.path.join(np_file_paths, "masks")
        # Sort so that each parallel read matches the corresponding .npy volume and mask
        self.image_files = sorted(os.listdir(self.imagesPath))
        self.mask_files = sorted(os.listdir(self.masksPath))
        assert len(self.image_files) == len(self.mask_files), "The number of images and masks must be equal"

        self.transform = transform

    def __getitem__(self, index):
        # Carica il volume e la maschera come array numpy
        X = np.load(os.path.join(self.imagesPath, self.image_files[index]))
        y = np.load(os.path.join(self.masksPath, self.mask_files[index]))

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        # Applica le trasformazioni se definite
        if self.transform:
            X_tensor, y_tensor = self.transform(X_tensor, y_tensor)
        
        return X_tensor, y_tensor
    
    def __len__(self):
        return len(self.image_files)