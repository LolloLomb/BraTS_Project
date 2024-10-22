import lightning as pl
import torch.optim as optim
from resnet import ResidualUNet3D
from torchmetrics.classification import Dice, MulticlassF1Score
import torch.nn as nn

class ResidualUNet3D_Lightning(pl.LightningModule):
    def __init__(self, in_channels, out_channels, loss_fn = None):
        super().__init__()
        self.model = ResidualUNet3D(in_channels=3, out_channels=4)
        self.loss_fn = nn.CrossEntropyLoss()
        self.dice_metric = MulticlassF1Score(average='none', num_classes=4)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        
        # Log del training loss
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        
        # Log delle metriche di validazione
        dice_score = self.dice_metric(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice_score, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
