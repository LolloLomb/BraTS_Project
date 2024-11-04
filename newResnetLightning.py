from newResnet import ResNet
import lightning as pl
import torch.optim as optim
import torch.nn as nn
import metrics as m
from metrics import Metrics
import torch

class ResNetLightning(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        torch.set_float32_matmul_precision('medium')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.model = ResNet(in_channels=in_channels, out_channels=out_channels).to('cuda')
        self.apply(self.init_weights)

        self.jaccard_step = []
        self.jaccard_epoch = []

        self.total_loss_step = []
        self.total_loss_epoch = []

        self.f1score_step = []
        self.f1score_epoch = []

        self.dice_score_step = []
        self.dice_score_epoch = []

        #self.hausdorff_step = []
        #self.hausdorff_epoch = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images.to('cuda')
        labels.to('cuda')
        predictions = self(images)
        
        labels = labels.permute(0, 4, 1, 2, 3)

        # Calcolo della loss
        total_loss = Metrics.combined_loss(predictions, labels)

        # Log
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images.to('cuda')
        labels.to('cuda')
        predictions = self(images)

        labels = labels.permute(0, 4, 1, 2, 3)

        # Calcolo delle metriche
        dice = Metrics.dice_coefficient(predictions, labels)
        jaccard = Metrics.jaccard_index(predictions, labels)
        f1score = Metrics.f1_score(predictions, labels)
        total_loss = Metrics.combined_loss(predictions, labels)

        # Accumula i valori per l'epoca corrente
        if self.trainer.state.stage != "sanity_check":
            self.jaccard_step.append(jaccard)
            self.f1score_step.append(f1score)
            self.total_loss_step.append(total_loss)
            self.dice_score_step.append(dice)

        # Log delle metriche
        #self.log('val_dice', dice, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('val_jaccard', jaccard, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('val_f1score', f1score, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss
    
    def on_validation_epoch_end(self):
        # Calcolo delle medie alla fine dell'epoca
        if self.trainer.state.stage != "sanity_check":
            print("\n\n")
            current_jaccards = []
            current_f1score = []
            current_dice_score = []
            for i in range(self.out_channels):
                s = sum(lst[i] for lst in self.jaccard_step)
                avg = len(self.jaccard_step)
                current_jaccards.append(s/avg)
            self.jaccard_epoch.append(current_jaccards)
            print("Jaccard: ", self.jaccard_epoch)

            for i in range(self.out_channels):
                s = sum(lst[i] for lst in self.f1score_step)
                avg = len(self.f1score_step)
                current_f1score.append(s/avg)
            self.f1score_epoch.append(current_f1score)
            
            print("Score: ", self.f1score_epoch)

            for i in range(self.out_channels):
                s = sum(lst[i] for lst in self.dice_score_step)
                avg = len(self.dice_score_step)
                current_dice_score.append(s/avg)
            self.dice_score_epoch.append(current_dice_score)
            
            print("Dice Coefficient: ", self.dice_score_epoch)

            avg_loss = sum(self.total_loss_step) / len(self.total_loss_step)
            self.total_loss_epoch.append(avg_loss)

            print("Loss: ", self.total_loss_epoch)
            
            self.total_loss_step.clear()
            self.jaccard_step.clear()
            self.f1score_step.clear()
            self.dice_score_step.clear()
            self.free_memory()

    def free_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_end(self):
        '''
        # Concatena le predizioni e le etichette accumulate
        all_predictions = torch.cat(self.all_predictions)
        all_labels = torch.cat(self.all_labels)

        # Calcola la Hausdorff Distance
        hausdorff = Metrics.hausdorff_distance(all_predictions, all_labels)

        # Log della Hausdorff Distance finale
        avg_hausdorff = hausdorff.mean().item()  # Media della distanza di Hausdorff
        self.hausdorff_epoch.append(avg_hausdorff)
        self.log('final_hausdorff', avg_hausdorff, on_epoch=True)

        # Stampa della Hausdorff Distance per debug
        print(f"\nFinal Hausdorff Distance: {avg_hausdorff}\n")

        # Pulisci i risultati accumulati
        self.all_predictions.clear()
        self.all_labels.clear()
        '''

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')