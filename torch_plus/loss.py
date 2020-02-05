import torch 
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(torch.nn.Module):
      def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            euclidean_distance = F.pairwise_distance(output1, output2) #can change distance metric 
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
            return loss_contrastive


class DepthAdjustedLoss(torch.nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss(reduction='none')):
        super(DepthAdjustedLoss, self).__init__()
        self.loss = loss

    def forward(self, *args, depths=None):
        adapted_loss = self.loss(*args)
        if (depths is not None):
            depths = depths.type(torch.FloatTensor)
            adapted_loss = torch.mul(depths,adapted_loss)
            adapted_loss = torch.mean(adapted_loss)
        return adapted_loss

