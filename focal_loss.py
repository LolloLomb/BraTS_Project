import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Calcola la BCE (Binary Cross Entropy)
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calcola la probabilità
        pt = torch.exp(-BCE_loss)

        # Calcola la Focal Loss
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        # Applica alpha se specificato
        if self.alpha is not None:
            # Assicurati che alpha sia un tensor
            alpha_tensor = torch.tensor(self.alpha).to(inputs.device)
            targets = targets.long()
            alpha_weights = (alpha_tensor).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            F_loss = F_loss * alpha_weights  # Applica il peso della classe
       
        loss_per_sample = F_loss.mean(dim=(2,3,4))
        loss_per_class = loss_per_sample.mean(dim=0)

        return loss_per_class
