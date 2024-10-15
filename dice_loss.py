import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, classes=4, weights=None, smooth=1e-7):
        super().__init__()
        self.classes = classes  # Number of classes for segmentation
        self.weights = weights  # Optional weights for each class
        self.smooth = smooth  # Smoothing factor to prevent division by zero

        # If weights are provided, check that they match the number of classes
        if weights is not None:
            if len(weights) != classes:
                raise ValueError("The number of classes must be equal to the length of the weights")

    def forward(self, inputs, targets):
        # Apply softmax to predicted inputs along the class channel
        inputs = F.softmax(inputs, dim=1)

        loss = 0.0  # Initialize loss to zero
        # Calculate Dice Loss for each class
        for class_idx in range(self.classes):
            # Use provided weight for the current class if available
            class_weight = self.weights[class_idx] if self.weights is not None else 1.0
            
            # Extract the predicted probabilities for the current class
            class_input = inputs[:, class_idx, :, :, :]  # Predictions for the current class

            # Directly use the one-hot encoded target for the current class
            class_target = targets[:, class_idx, :, :, :]  # One-hot mask for the current class

            # Flatten the inputs and targets to simplify the calculation
            class_input_flat = class_input.contiguous().view(-1)
            class_target_flat = class_target.contiguous().view(-1)

            # Calculate the intersection and union for the Dice Loss formula
            intersection = torch.sum(class_input_flat * class_target_flat)  # Intersection of predicted and target
            union = torch.sum(class_input_flat) + torch.sum(class_target_flat)  # Union of predicted and target
            
            # Compute Dice Loss for the current class
            class_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

            # Add the weighted loss for the current class to the total loss
            loss += class_loss * class_weight

        # Return the average loss across all classes
        return loss / self.classes
