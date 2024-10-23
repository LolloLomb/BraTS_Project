from scipy.spatial.distance import directed_hausdorff
import numpy as np
import torch

lossWeights = [ 0.001636661,
                0.685754189,
                0.065468574,
                0.247140576]

class Metrics:
    @staticmethod
    def dice_loss(prediction, target, weights=lossWeights, epsilon=1e-6):
        """
        Calcola la Dice Loss pesata
        """
        # Inizializza la loss totale
        total_loss = 0.0

        # Calcola la Dice Loss per ogni classe
        for i in range(prediction.shape[1]):  # Itera su ciascuna classe
            # Calcola l'intersezione tra predizione e target per la classe i
            intersection = (prediction[:, i] * target[:, i]).sum()
    
            # Calcola il coefficiente di Dice
            dice_coeff = (2. * intersection + epsilon) / (prediction[:, i].sum() + target[:, i].sum() + epsilon)
    
            # Applica il peso se fornito
            if weights is not None:
                total_loss += weights[i] * (1 - dice_coeff)
            else:
                total_loss += (1 - dice_coeff)
    
        return total_loss

    @staticmethod
    def focal_loss(prediction, target, weights=None, alpha=1.0, gamma=2.0, epsilon=1e-6):
        """
        Calcola la Focal Loss pesata.
        """
        prediction = prediction.clamp(epsilon, 1 - epsilon)  # Clamping per stabilità numerica
        target = target.float()  # Assicurati che il target sia in formato float

        total_focal_loss = 0.0
        
        # Inizializza la Focal Loss totale
        for i in range(prediction.shape[1]):  # Itera su ciascuna classe
            p_t = prediction[:, i]  # Probabilità per la classe i
            t = target[:, i]  # Target (one-hot) per la classe i
            
            # Calcolo della Focal Loss per la classe i
            focal_loss_value = -alpha * (1 - p_t) ** gamma * t * p_t.log()  # Focal loss per i positivi
            focal_loss_value_neg = -(1 - alpha) * p_t ** gamma * (1 - t) * (1 - p_t).log()  # Per i negativi
    
            focal_loss = focal_loss_value + focal_loss_value_neg
            
            # Applica il peso se fornito
            if weights is not None:
                total_focal_loss += weights[i] * focal_loss.mean()
            else:
                total_focal_loss += focal_loss.mean()
    
        return total_focal_loss

    @staticmethod
    def combined_loss(prediction, target, weights=lossWeights):
        """
        Combina Dice Loss e Focal Loss in una singola loss.
        """
        dice = Metrics.dice_loss(prediction, target, weights)
        focal = Metrics.focal_loss(prediction, target, weights)
        
        return 0.5 * dice + 0.5 * focal  # Combina le due perdite

    @staticmethod
    def hausdorff_distance(prediction, target):
        """
        Calcola la Hausdorff Distance tra prediction e target
        per ciascuna classe e per ciascun volume nel batch.
        """
        # Assicurati che prediction e target siano sulla GPU
        device = prediction.device  # Ottieni il dispositivo delle predizioni
        batch_size, num_classes, d, h, w = prediction.shape
        
        # Inizializza le distanze come infinito
        distances = torch.full((batch_size, num_classes), float('inf'), device=device)  
        
        for b in range(batch_size):
            for c in range(num_classes):
                # Trova le coordinate dei voxel non zero
                pred_coords = torch.nonzero(prediction[b, c]).to(device)  # (num_voxel, 3)
                target_coords = torch.nonzero(target[b, c]).to(device)  # (num_voxel, 3)
    
                # Debug per verificare le dimensioni delle coordinate
                print(f'Batch {b}, Classe {c}: pred_coords shape = {pred_coords.shape}, target_coords shape = {target_coords.shape}')
                
                # Se non ci sono voxel per la classe c in predizione o target, skip
                if pred_coords.size(0) == 0 or target_coords.size(0) == 0:
                    print(f"No valid voxels for Batch {b}, Class {c}, keeping distance as infinity.")
                    continue  # Lascia infinito se ci sono voxel non validi
    
                # Calcola la Hausdorff Distance diretta e inversa usando numpy
                pred_coords_np = pred_coords.cpu().numpy()  # Converti in numpy per scipy
                target_coords_np = target_coords.cpu().numpy()
    
                # Calcola le distanze di Hausdorff
                dist1 = directed_hausdorff(pred_coords_np, target_coords_np)[0]
                dist2 = directed_hausdorff(target_coords_np, pred_coords_np)[0]
                
                # Prendi la distanza massima tra i due
                hausdorff_dist = max(dist1, dist2)
                distances[b, c] = hausdorff_dist  # Assegna la distanza calcolata
                
        # Restituisci le distanze per batch e classe come un array sulla GPU
        return distances
                
    @staticmethod
    def dice_coefficient(prediction, target, epsilon=1e-6):
        """
        Calcola il coefficiente di Dice.
        """
        intersection = (prediction * target).sum()
        return (2. * intersection + epsilon) / (prediction.sum() + target.sum() + epsilon)

    @staticmethod
    def accuracy(prediction, target):
        """
        Calcola l'accuratezza classica.
        """
        preds = (prediction > 0.5).float()  # Applicazione di una soglia
        correct = (preds == target).float().sum()
        return correct / target.numel()

    @staticmethod
    def jaccard_index(prediction, target, epsilon=1e-6):
        """
        Calcola l'indice di Jaccard.
        """
        intersection = (prediction * target).sum()
        union = prediction.sum() + target.sum() - intersection
        return (intersection + epsilon) / (union + epsilon)


