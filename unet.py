import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, JaccardIndex

torch.cuda.empty_cache()

# Convolutional Block class
# Defines a block consisting of two 3D convolutional layers with ReLU activations and dropout.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding='same')
        self.drop = nn.Dropout(dropout)  # Dropout for regularization
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding='same')
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.conv1(x))  # First convolution followed by ReLU
        x = self.drop(x)  # Dropout after first convolution
        x = self.relu(self.conv2(x))  # Second convolution followed by ReLU
        return x
    

# Encoder Block class
# This block performs downsampling using a convolutional block followed by max pooling.
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # Pooling for downsampling
    
    def forward(self, x):
        x = self.conv_block(x)  # Apply convolutional block
        p = self.pool(x)  # Perform max pooling
        return x, p  # Return both the feature map and the pooled output


# Decoder Block class
# Performs upsampling using transposed convolution and then concatenates the feature maps with skip connections.
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)  # Upsample
        self.conv_block = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.upconv(x)  # Upsample
        x = torch.cat((x, skip), dim=1)  # Concatenate with the skip connection
        x = self.conv_block(x)  # Apply convolutional block
        return x


# UNet 3D model class
# Implements a 3D version of the UNet architecture for volumetric data, useful for tasks like brain tumor segmentation.
class UNet3D(L.LightningModule):

    def __init__(self, in_channels, out_channels, loss_fx, learning_rate):
        super().__init__()
        self.loss_fx = loss_fx
        self.learning_rate = learning_rate

        # Define the encoder blocks
        self.encoder1 = EncoderBlock(in_channels, 16, 0.1)
        self.encoder2 = EncoderBlock(16, 32, 0.1)
        self.encoder3 = EncoderBlock(32, 64, 0.2)
        self.encoder4 = EncoderBlock(64, 128, 0.2)

        # Bottleneck layer for the lowest resolution representation
        self.bottleneck = ConvBlock(128, 256, 0.3)

        # Define the decoder blocks for upsampling
        self.decoder4 = DecoderBlock(256, 128, 0.2)
        self.decoder3 = DecoderBlock(128, 64, 0.2)
        self.decoder2 = DecoderBlock(64, 32, 0.1)
        self.decoder1 = DecoderBlock(32, 16, 0.1)

        # Final convolutional layer to output the segmentation mask
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)

        # Initialize model weights
        self.apply(self.init_weights)

        # Metrics: accuracy and Jaccard index for both training and validation
        self.train_accuracy = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=out_channels)
        self.train_jaccard = JaccardIndex(task='multiclass', num_classes=out_channels)
        self.val_jaccard = JaccardIndex(task='multiclass', num_classes=out_channels)

        # Lists to store metrics
        self.dice_values = []
        self.val_accuracy_values = []
        self.jaccard_values = []

    # Forward pass of the model
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # Transpose to [batch_size, channels, depth, height, width]
        # Encoder path
        x1, p1 = self.encoder1(x)
        x2, p2 = self.encoder2(p1)
        x3, p3 = self.encoder3(p2)
        x4, p4 = self.encoder4(p3)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder path (upsampling with skip connections)
        d4 = self.decoder4(bn, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        # Apply softmax to the final convolutional layer
        outputs = torch.softmax(self.final_conv(d1), dim=1)
        return outputs

    # Training step (per batch)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        y = y.permute(0, 4, 1, 2, 3)  # Cambia l'ordine delle dimensioni
        loss = self.loss_fx(y_out, y)
        
        # Update and log metrics
        self.train_accuracy(y_out, y)
        self.train_jaccard(y_out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_jaccard', self.train_jaccard, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    # Validation step (per batch)
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        y = y.permute(0, 4, 1, 2, 3)  # Cambia l'ordine delle dimensioni
        loss = self.loss_fx(y_out, y)
        
        # Update and log metrics
        self.val_accuracy(y_out, y)
        self.val_jaccard(y_out, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_jaccard', self.val_jaccard, on_step=True, on_epoch=True, prog_bar=True)

        # Save values at the end of validation step
        self.val_accuracy_values.append(self.val_accuracy.compute())  # Append accuracy
        self.jaccard_values.append(self.val_jaccard.compute())      # Append Jaccard index
        self.dice_values.append(loss.item())                        # Append loss (Dice + Focal)

        # Reset metrics for next validation step
        self.val_accuracy.reset()
        self.val_jaccard.reset()

        print("Val Accuracy ", self.val_accuracy_values)
        print("Dice ", self.dice_values)
        print("Jaccard ", self.jaccard_values)

        return loss

    # Count the total number of parameters in the model
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 

    # Weight initialization function
    def init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # Configure the optimizer
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optim
