from scipy.spatial.distance import directed_hausdorff
import numpy as np
import torch

lossWeights = [ 0.001636661,
                0.685754189,
                0.065468574,
                0.247140576]

class Metrics:

    @staticmethod
    def precision(pred_class, target_class, epsilon=1e-6):
        """
        Calculates Precision for a specific class in segmentation.
        """
    
        pred_class = (pred_class > 0.5).float()  # Binarize the prediction
        target_class = target_class.float()
        
        # Calculate true positive and predicted positive
        true_positive = (pred_class * target_class).sum()
        predicted_positive = pred_class.sum()
        
        # Calculate precision
        class_precision = (true_positive + epsilon) / (predicted_positive + epsilon)
        class_precision = class_precision.item()
        
        return class_precision  # Return precision


    @staticmethod
    def recall(pred_class, target_class, epsilon=1e-6):
        """
        Calculates Recall for a specific class in segmentation.
        """
        pred_class = (pred_class > 0.5).float()  # Binarize the prediction
        target_class = target_class.float()
        
        # Calculate true positive and actual positive
        true_positive = (pred_class * target_class).sum()
        actual_positive = target_class.sum()
        
        # Calculate recall
        class_recall = (true_positive + epsilon) / (actual_positive + epsilon)
        class_recall = class_recall.item()
        
        return class_recall  # Return recall

    @staticmethod
    def f1_score(prediction, target, epsilon=1e-6):
        """
        Calculates F1-Score for segmentation.
        """
        num_classes = prediction.shape[1]
            
        class_f1_scores = []
    
        # Calculate precision and recall for each class
        for class_idx in range(num_classes):
            pred_class = prediction[:, class_idx, ...]
            target_class = target[:, class_idx, ...]
    
            # Calculate precision and recall for the current class
            precision_value = Metrics.precision(pred_class, target_class)
            recall_value = Metrics.recall(pred_class, target_class)
    
            # Calculate F1-Score for the current class
            f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + epsilon)
    
            # Apply the weight to the current class (if needed)
            class_f1_scores.append(f1)
        
        # Return the F1 scores for each class
        return class_f1_scores

    @staticmethod
    def dice_loss(prediction, target, weights=lossWeights, epsilon=1e-6):
        """
        Calculates the weighted Dice Loss.
        """
        # Initialize the total loss
        total_loss = 0.0

        # Calculate Dice Loss for each class
        for i in range(prediction.shape[1]):  # Iterate over each class
            # Calculate the intersection between prediction and target for class i
            intersection = (prediction[:, i] * target[:, i]).sum()
    
            # Calculate the Dice coefficient
            dice_coeff = (2. * intersection + epsilon) / (prediction[:, i].sum() + target[:, i].sum() + epsilon)
    
            # Apply the weight if provided
            if weights is not None:
                total_loss += weights[i] * (1 - dice_coeff)
            else:
                total_loss += (1 - dice_coeff)
    
        return total_loss

    @staticmethod
    def focal_loss(prediction, target, weights=lossWeights, alpha=1, gamma=2.0, epsilon=1e-6):
        """
        Calculates the weighted Focal Loss.
        """
        prediction = prediction.clamp(epsilon, 1 - epsilon)  # Clamping for numerical stability
        target = target.float()  # Ensure the target is in float format

        total_focal_loss = 0.0
        
        # Initialize the total Focal Loss
        for i in range(prediction.shape[1]):  # Iterate over each class
            p_t = prediction[:, i]  # Probability for class i
            t = target[:, i]  # Target (one-hot) for class i
            
            # Calculate Focal Loss for class i
            focal_loss_value = -alpha * (1 - p_t) ** gamma * t * p_t.log()  # Focal loss for positives
            focal_loss_value_neg = -(1 - alpha) * p_t ** gamma * (1 - t) * (1 - p_t).log()  # For negatives
    
            focal_loss = focal_loss_value + focal_loss_value_neg
            
            # Apply the weight if provided
            if weights is not None:
                total_focal_loss += weights[i] * focal_loss.mean()
            else:
                total_focal_loss += focal_loss.mean()
    
        return total_focal_loss

    @staticmethod
    def combined_loss(prediction, target, weights=lossWeights):
        """
        Combines Dice Loss and Focal Loss into a single loss.
        """
        dice = Metrics.dice_loss(prediction, target, weights)
        focal = Metrics.focal_loss(prediction, target, weights)
        
        return 0.4 * dice + 0.6 * focal  # Combine the two losses

    @staticmethod
    def hausdorff_distance(prediction, target):
        """
        Calculates the Hausdorff Distance between prediction and target
        for each class and for each volume in the batch.
        """
        # Ensure prediction and target are on the GPU
        device = prediction.device  # Get the device of the predictions
        batch_size, num_classes, d, h, w = prediction.shape
        
        # Initialize distances as infinity
        distances = torch.full((batch_size, num_classes), float('inf'), device=device)  
        
        for b in range(batch_size):
            for c in range(num_classes):
                # Find coordinates of non-zero voxels
                pred_coords = torch.nonzero(prediction[b, c]).to(device)  # (num_voxel, 3)
                target_coords = torch.nonzero(target[b, c]).to(device)  # (num_voxel, 3)
    
                # Debug to verify coordinate dimensions
                print(f'Batch {b}, Class {c}: pred_coords shape = {pred_coords.shape}, target_coords shape = {target_coords.shape}')
                
                # If there are no voxels for class c in prediction or target, skip
                if pred_coords.size(0) == 0 or target_coords.size(0) == 0:
                    print(f"No valid voxels for Batch {b}, Class {c}, keeping distance as infinity.")
                    continue  # Leave distance as infinity if there are no valid voxels
    
                # Calculate directed and inverse Hausdorff Distances using numpy
                pred_coords_np = pred_coords.cpu().numpy()  # Convert to numpy for scipy
                target_coords_np = target_coords.cpu().numpy()
    
                # Calculate Hausdorff distances
                dist1 = directed_hausdorff(pred_coords_np, target_coords_np)[0]
                dist2 = directed_hausdorff(target_coords_np, pred_coords_np)[0]
                
                # Take the maximum distance between the two
                hausdorff_dist = max(dist1, dist2)
                distances[b, c] = hausdorff_dist  # Assign the calculated distance
                
        # Return distances for batch and class as an array on the GPU
        return distances
                
    @staticmethod
    def dice_coefficient(prediction, target, weights=lossWeights, epsilon=1e-6):
        """
        Calcola il coefficiente di Dice pesato.
        """
        num_classes = prediction.shape[1]

        # Se i pesi non sono specificati, assegna un peso uguale a tutte le classi
        if weights is None:
            weights = [1.0] * num_classes

        class_dice_scores = []

        # Calcola il coefficiente di Dice per ciascuna classe
        for class_idx in range(num_classes):
            pred_class = prediction[:, class_idx, ...]
            target_class = target[:, class_idx, ...]

            # Calcola l'intersezione e la somma delle previsioni e dei target per la classe corrente
            intersection = (pred_class * target_class).sum()
            dice_score = (2. * intersection + epsilon) / (pred_class.sum() + target_class.sum() + epsilon)

            # Applica il peso per la classe corrente
            class_dice_scores.append(dice_score * weights[class_idx])

        # Restituisce la somma dei coefficienti di Dice pesati per ciascuna classe
        return sum(class_dice_scores) / sum(weights)

    '''
    @staticmethod
    def accuracy(prediction, target):
        """
        Calculates classic accuracy.
        """
        preds = (prediction > 0.5).float()  # Apply a threshold
        correct = (preds == target).float().sum()
        return correct / target.numel()
    '''

    
    @staticmethod
    def jaccard_index(prediction, target, weights=lossWeights, epsilon=1e-6):
        num_classes = prediction.shape[1]
        
        # If weights are not specified, assign equal weight to all classes
        if weights is None:
            weights = [1.0] * num_classes
    
        class_jaccard_scores = []
    
        # Calculate Jaccard index for each class
        for class_idx in range(num_classes):
            pred_class = prediction[:, class_idx, ...]
            target_class = target[:, class_idx, ...]
    
            # Calculate intersection and union
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum() - intersection
    
            # Calculate Jaccard for the current class
            jaccard = (intersection + epsilon) / (union + epsilon)
            jaccard = jaccard.item()
    
            # Apply the weight to the current class (if needed)
            class_jaccard_scores.append(jaccard * weights[class_idx])
    
        # Return the Jaccard scores for each class
        return class_jaccard_scores
