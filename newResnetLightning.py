from newResnet import ResNet
import lightning as pl
import torch.optim as optim
import metrics as m
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ResNetLightning(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = ResNet(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        images, labels = batch
        predictions = self(images)
        print(predictions.shape, labels.shape)
        # Calcolo della loss

        # Log

    def validation_step(self, batch):
        images, labels = batch
        predictions = self(images)

        # Calcolo della loss

        # Log
    
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
        

        
        