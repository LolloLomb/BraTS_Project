import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        
        # Prima convoluzione 3D
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Seconda convoluzione 3D
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU()
        
        # Percorso shortcut (connessione residua)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        # Percorso principale
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Aggiunta della connessione residua
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Encoder3D(nn.Module):
    def __init__(self, in_channels):
        super(Encoder3D, self).__init__()
        self.layer1 = ResidualBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer2 = ResidualBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer3 = ResidualBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer4 = ResidualBlock3D(128, 256)

    def forward(self, x):
        x1 = self.layer1(x)  # First block output
        x2 = self.layer2(self.pool1(x1))  # Downsample and process
        x3 = self.layer3(self.pool2(x2))
        x4 = self.layer4(self.pool3(x3))
        return x1, x2, x3, x4
    
class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()
        
        # Upsampling layers with transposed convolutions
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.layer3 = ResidualBlock3D(256, 128)
        
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.layer2 = ResidualBlock3D(128, 64)
        
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.layer1 = ResidualBlock3D(64, 32)
        
        # Output layer
        self.conv_out = nn.Conv3d(32, 1, kernel_size=1)
    
    def forward(self, x1, x2, x3, x4):
        d3 = self.layer3(torch.cat([self.upconv3(x4), x3], dim=1))
        d2 = self.layer2(torch.cat([self.upconv2(d3), x2], dim=1))
        d1 = self.layer1(torch.cat([self.upconv1(d2), x1], dim=1))
        return self.conv_out(d1)

class ResidualUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResidualUNet3D, self).__init__()
        self.encoder = Encoder3D(in_channels)
        self.decoder = Decoder3D()
    
    def forward(self, x):
        # Passaggio attraverso l'encoder
        x = x.permute(0, 4, 1, 2, 3)
        x1, x2, x3, x4 = self.encoder(x)
        
        # Passaggio attraverso il decoder
        out = self.decoder(x1, x2, x3, x4)
        
        return out
