import torch
import torch.nn as nn
import torch.nn.functional as F

def correlation_aware_loss(pred, gt, mask):
    """
    Calculates the correlation-aware loss, which maximizes the Pearson correlation 
    between predicted and ground-truth stock returns.
    """
    valid_mask = mask.bool().squeeze()
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    pred_std = torch.std(pred_valid) + 1e-8
    gt_std = torch.std(gt_valid) + 1e-8

    pred_normalized = (pred_valid - torch.mean(pred_valid)) / pred_std
    gt_normalized = (gt_valid - torch.mean(gt_valid)) / gt_std

    ic = torch.mean(pred_normalized * gt_normalized)

    return 1 - ic 

def get_loss(prediction, ground_truth, base_price, mask, batch_size, reg_loss_weight, rank_loss_weight, ic_weight):
   """
   Calculates the combined loss function for stock trend prediction.
   Combines regression loss (Huber) and correlation-aware loss.
   """
   device = prediction.device

   return_ratio = torch.div(torch.sub(prediction, base_price), base_price)

   huber_loss_fn = nn.HuberLoss(delta=1.0).to(device)
   reg_loss = huber_loss_fn(return_ratio * mask, ground_truth * mask)

   ic_loss = correlation_aware_loss(return_ratio, ground_truth, mask)

   total_loss = reg_loss_weight * reg_loss + ic_weight * ic_loss 

   return total_loss, return_ratio