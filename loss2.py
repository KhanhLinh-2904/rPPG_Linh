import torch
import torch.nn as nn
import torch.nn.modules.loss as loss

def loss_fn(loss_fn):
    if loss_fn == 'combined_loss':
        return Combined_Loss()
  
######################################################3
def neg_Pearson_Loss_MTTS(predictions, targets):
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = torch.squeeze(predictions)
    targets = torch.squeeze(targets)
    if len(predictions.shape) >= 2:
        predictions = predictions.view(-1)
    if len(targets.shape) >= 2:
        targets = targets.view(-1)
    sum_x = torch.sum(predictions)  # x
    sum_y = torch.sum(targets)  # y
    sum_xy = torch.sum(predictions * targets)  # xy
    sum_x2 = torch.sum(torch.pow(predictions, 2))  # x^2
    sum_y2 = torch.sum(torch.pow(targets, 2))  # y^2
    t = len(predictions)
    pearson = (t * sum_xy - (sum_x * sum_y)) / (torch.sqrt((t * sum_x2 - torch.pow(sum_x, 2)) * (t * sum_y2 - torch.pow(sum_y, 2))))

    return 1 - pearson
###########################################################################################



#########################################################
class NegPearsonLoss_MTTS(nn.Module):
    def __init__(self):
        super(NegPearsonLoss_MTTS, self).__init__()

    def forward(self, predictions, targets):
        return neg_Pearson_Loss_MTTS(predictions, targets)
############################################################

class Combined_Loss(nn.Module):
    def __init__(self):
        super(Combined_Loss, self).__init__()
        self.mse = loss.MSELoss(reduction='mean')
        self.pearson = NegPearsonLoss_MTTS()

    def forward(self, predictions, targets):
        mse = self.mse(predictions, targets)
        pearson = self.pearson(predictions, targets)
        total = mse + pearson     # change this
        # total = mse
        return total
