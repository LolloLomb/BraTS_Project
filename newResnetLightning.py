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

        self.dice_values_step = []
        self.val_accuracy_values_step = []
        self.jaccard_values_step = []
        self.dice_values = []
        self.val_accuracy_values = []
        self.jaccard_values = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        images, labels = batch
        predictions = self(images)
        
        labels = labels.permute(0, 4, 1, 2, 3)

        # Calcolo della loss
        total_loss = Metrics.combined_loss(predictions, labels)
        # Log
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch):
        images, labels = batch
        predictions = self(images)
        # Dentro predictions ci finisce il risultato della softmax su quel batch, devo debuggare
        labels = labels.permute(0, 4, 1, 2, 3)
        
        # Calcolo della loss
        dice = Metrics.dice_coefficient(predictions, labels)
        #hausdorff = Metrics.hausdorff_distance(predictions, labels)
        acc = Metrics.accuracy(predictions, labels)
        jaccard = Metrics.jaccard_index(predictions, labels)
        total_loss = Metrics.combined_loss(predictions, labels)

        # Log
        self.log('val_dice', dice)
        #self.log('val_hausdorff', hausdorff)
        self.log('val_accuracy', acc)
        self.log('val_jaccard', jaccard)
        self.log('val_loss', total_loss)
        
        # Save values at the end of validation step
        if self.trainer.state.stage != "sanity_check":
            self.val_accuracy_values_step.append(acc)  # Append accuracy
            self.jaccard_values_step.append(jaccard)      # Append Jaccard index
            self.dice_values_step.append(total_loss)                        # Append loss (Dice + Focal loss)

        return total_loss
    
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

    def on_validation_epoch_end(self):
        if self.trainer.state.stage != "sanity_check":
            self.val_accuracy_values.append(sum(self.val_accuracy_values_step)/len(self.val_accuracy_values_step))
            self.jaccard_values.append(sum(self.jaccard_values_step)/len(self.jaccard_values_step))
            self.dice_values.append(sum(self.dice_values_step)/len(self.dice_values_step))

        self.val_accuracy_values_step.clear()
        self.jaccard_values_step.clear()
        self.dice_values_step.clear()

        print(f"Val Acc: {self.val_accuracy_values}, jaccard: {self.jaccard_values}, dice: {self.dice_values}")

        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        
        
