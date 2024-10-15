import nibabel as nib                           # Used to load the samples
import numpy as np                              # Necessary for converting to numpy arrays
from sklearn.preprocessing import MinMaxScaler  # To impose which class we want to accept as input for crop_and_save
# import matplotlib.pyplot as plt                 # For debugging and for obtaining the slices to place on the Blender cube

# Redefine the to_categorical function as it works for Keras but in Torch
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def crop_and_save(t2_list : list, t1ce_list : list, flair_list : list, mask_list : list, treshold : float, scaler : MinMaxScaler):
    
    # Load the image
    # Combine all channels except the last one
    # Pass the resulting 2D matrix to scaler.fit_transform, which will scale the results into float [0,1]
    # Return to the original dimensions
    # Repeat for each image

    for img in range(len(t2_list)): # All lists have the same length
        print("Now preparing image and masks number: ", img)

        temp_image_t2 = nib.load(t2_list[img]).get_fdata() # Returns a numpy array
        temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
        
        temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
        temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

        temp_image_flair = nib.load(flair_list[img]).get_fdata()
        temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

        temp_image_flair = temp_image_flair[56:184, 56:184, 13:141]

        # Load the mask
        # Convert values to integers
        # Since label 3 is no longer used, all pixels with label 4 now have label 3
        # I do this to maintain continuity and use only 0, 1, 2, 3 as labels

        temp_mask = nib.load(mask_list[img]).get_fdata()

        # Convert to numpy data
        temp_mask = temp_mask.astype(np.uint8)
        # Remove class 4 to maintain continuity
        temp_mask[temp_mask == 4] = 3

        # Combine images from the different scans, making them a single multi-channel volume
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

        # Perform cropping centered around the brain, extracting a 3D patch of 128x128x128 (since it's a multiple of 64)
        temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]

        temp_mask = temp_mask[56:184, 56:184, 13:141]

        # Inside `val` I get [0 1 2 3], while `counts` is the count of respective occurrences for each pixel in the image
        _, counts = np.unique(temp_mask, return_counts=True)

        # At least 1% of the volume must have a useful label
        if (1 - (counts[0]/counts.sum())) > treshold:

            # After the to_categorical function, the temp_mask has a new channel
            # 128x128x128 ---> 128x128x128x4 (where the class is highlighted)
            temp_mask = to_categorical(temp_mask, num_classes=4)

            np.save('new2024/input_data_128/train/images/image_'+str(img)+'.npy', temp_combined_images)
            np.save('new2024/input_data_128/train/masks/mask_'+str(img)+'.npy', temp_mask)
        # Otherwise, discard
        else:
            pass
