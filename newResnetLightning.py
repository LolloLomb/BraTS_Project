from newResnet import ResNet
import lightning as pl
import torch.optim as optim
import torch.nn as nn
import metrics as m
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import Metrics

class ResNetLightning(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = ResNet(in_channels=in_channels, out_channels=out_channels)
        self.apply(self.init_weights)


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        images, labels = batch
        predictions = self(images)
        # Calcolo della loss
        dice_loss = Metrics.dice_loss(predictions, labels)
        # Log
        self.log("train_loss", dice_loss)
        return dice_loss

    def validation_step(self, batch):
        images, labels = batch
        predictions = self(images)

        # Calcolo della loss
        dice = Metrics.dice_coefficient(predictions, labels)
        hausdorff = Metrics.hausdorff_distance(predictions, labels)
        acc = Metrics.accuracy(predictions, labels)
        jaccard = Metrics.jaccard_index(predictions, labels)
        dice_loss = Metrics.dice_loss(predictions, labels)

        # Log
        self.log('val_dice', dice)
        self.log('val_hausdorff', hausdorff)
        self.log('val_accuracy', acc)
        self.log('val_jaccard', jaccard)
        self.log('val_dice_loss', dice_loss)
    
    def configure_optimizers(self):
        # Prima definisco l'optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        # Definisco lo scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        
        