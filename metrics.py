from scipy.spatial.distance import directed_hausdorff

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
        
        # Assicurati che i tensori siano nella forma giusta
        assert prediction.dim() == target.dim() == 4, "Prediction and target must have 4 dimensions (N, C, H, W)"
        
        # Inizializza la loss totale
        total_loss = 0.0

        # Calcola la Dice Loss per ogni classe
        for i in range(prediction.shape[1]):  # Itera su ciascuna classe
            intersection = (prediction[:, i] * target[:, i]).sum()
            dice_coeff = (2. * intersection + epsilon) / (prediction[:, i].sum() + target[:, i].sum() + epsilon)

            # Applica il peso
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
        
        # Inizializza la Focal Loss totale
        total_focal_loss = 0.0
        
        # Calcola la Focal Loss per ogni classe
        for i in range(prediction.shape[1]):  # Itera su ciascuna classe
            p_t = prediction[:, i]  # Probabilità per la classe i
            t = target[:, i]  # Target per la classe i
            
            # Calcolo della Focal Loss per la classe i
            focal_loss_value = -alpha * (1 - p_t) ** gamma * t * p_t.log()
            # Applica il peso se fornito
            if weights is not None:
                total_focal_loss += weights[i] * focal_loss_value.mean()
            else:
                total_focal_loss += focal_loss_value.mean()

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
        Calcola la Hausdorff Distance.
        """
        pred_coords = prediction.nonzero(as_tuple=True)
        target_coords = target.nonzero(as_tuple=True)

        if pred_coords[0].numel() == 0 or target_coords[0].numel() == 0:
            return float('inf')  # Se non ci sono punti, restituisci infinito

        dist1 = directed_hausdorff(pred_coords, target_coords)[0]
        dist2 = directed_hausdorff(target_coords, pred_coords)[0]
        return max(dist1, dist2)

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


