import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        # applico i filtri 32, 64, 128, 256
        super(ResNet, self).__init__()

        self.inputLayer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1),
            nn.Dropout(0.1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1)
        )

        # Devo controllare le dimensioni di uscita da questo InputSkip per capire se posso sommare bene questo residuo
        self.inputSkip = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=1, padding=0)

        self.residualConv1 = ResConv(32, 64, 2, 1, dropout=0.2)
        self.residualConv2 = ResConv(64, 128, 2, 1, dropout=0.2)

        self.bridge = ResConv(128, 256, 2, 1, dropout=0.3)

        self.upsample1 = Upsample(256, 256, 2, 2)
        self.upResConv1 = ResConv(256 + 128, 128, 1, 1, dropout=0.2)

        self.upsample2 = Upsample(128, 128, 2, 2)
        self.upResConv2 = ResConv(128 + 64, 64, 1, 1, dropout=0.2)

        self.upsample3 = Upsample(64, 64, 2, 2)
        self.upResConv3 = ResConv(64 + 32, 32, 1, 1, dropout=0.1)

        self.outputLayer = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1)
        )


    
    def forward(self, x):
        # Fase di encoding
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.inputLayer(x) + self.inputSkip(x)
        x2 = self.residualConv1(x1)
        x3 = self.residualConv2(x2)

        # Ponte
        x4 = self.bridge(x3)

        # Fase di decoding
        x4 = self.upsample1(x4)
        x5 = torch.cat((x4, x3), dim=1)
        x6 = self.upResConv1(x5)

        x6 = self.upsample2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.upResConv2(x7)

        x8 = self.upsample3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.upResConv3(x9)

        x10 = self.outputLayer(x10)

        probabilities = torch.softmax(x10, dim=1)

        return probabilities


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channles, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channles, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)



class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, dropout):
        super(ResConv, self).__init__()

        self.convBlock = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3,3), stride=stride, padding=padding),
            nn.Dropout(dropout),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3,3), stride=1, padding=1)
        )

        self.convSkip = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.convBlock(x) + self.convSkip(x)




