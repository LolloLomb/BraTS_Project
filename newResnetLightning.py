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
        
        self.model = ResNet(in_channels=in_channels, out_channels=out_channels)
        self.apply(self.init_weights)

        self.jaccard_step = []
        self.jaccard_epoch = []

        self.total_loss_step = []
        self.total_loss_epoch = []

        self.f1score_step = []
        self.f1score_epoch = []

        self.precision_step = []
        self.precision_epoch = []
        self.recall_step = []
        self.recall_epoch = []

        self.hausdorff_step = []
        self.hausdorff_epoch = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        
        labels = labels.permute(0, 4, 1, 2, 3)

        # Calcolo della loss
        total_loss = Metrics.combined_loss(predictions, labels)

        # Log
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)

        labels = labels.permute(0, 4, 1, 2, 3)

        # Calcolo delle metriche
        dice = Metrics.dice_coefficient(predictions, labels)
        jaccard = Metrics.jaccard_index(predictions, labels)
        precision = Metrics.precision(predictions, labels)
        recall = Metrics.recall(predictions, labels)
        f1score = Metrics.f1_score(predictions, labels)
        total_loss = Metrics.combined_loss(predictions, labels)

        # Accumula i valori per l'epoca corrente
        if self.trainer.state.stage != "sanity_check":
            self.jaccard_step.append(jaccard.item())
            self.precision_step.append(precision.item())
            self.recall_step.append(recall.item())
            self.f1score_step.append(f1score.item())
            self.total_loss_step.append(total_loss.item())

        # Log delle metriche
        self.log('val_dice', dice, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_jaccard', jaccard, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_f1score', f1score, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss
    
    def on_validation_epoch_end(self):
        # Calcolo delle medie alla fine dell'epoca
        if self.trainer.state.stage != "sanity_check":
            avg_jaccard = sum(self.jaccard_step) / len(self.jaccard_step)
            avg_precision = sum(self.precision_step) / len(self.precision_step)
            avg_recall = sum(self.recall_step) / len(self.recall_step)
            avg_f1score = sum(self.f1score_step) / len(self.f1score_step)
            avg_loss = sum(self.total_loss_step) / len(self.total_loss_step)
    
            # Aggiungi le medie alle liste dell'epoca
            self.jaccard_epoch.append(avg_jaccard)
            self.precision_epoch.append(avg_precision)
            self.recall_epoch.append(avg_recall)
            self.f1score_epoch.append(avg_f1score)
            self.total_loss_epoch.append(avg_loss)

            # Pulisci i valori batch-wise dopo aver calcolato la media
            self.jaccard_step.clear()
            self.precision_step.clear()
            self.recall_step.clear()
            self.f1score_step.clear()
            self.total_loss_step.clear()
    
            # Stampa dei valori medi per debug
            print(f"\nVal Jaccard: {avg_jaccard}\n"
                  f"Val Precision: {avg_precision}\n"
                  f"Val Recall: {avg_recall}\n"
                  f"Val F1-Score: {avg_f1score}\n"
                  f"Val Loss: {avg_loss}\n")
    
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