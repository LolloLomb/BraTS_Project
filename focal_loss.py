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
            F_loss = F_loss * alpha_tensor[targets]  # Applica il peso della classe

        return F_loss.mean()
