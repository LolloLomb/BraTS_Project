import torch
import torch.nn

class CombinedLoss(torch.nn.Module):
    def __init__(self, dice_loss, focal_loss, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = dice_loss
        self.focal_loss = focal_loss
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        # Calcola la Dice Loss e la Focal Loss
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)

        # Combina le due perdite usando i pesi
        combined_loss = self.dice_weight * dice + self.focal_weight * focal
        return combined_loss