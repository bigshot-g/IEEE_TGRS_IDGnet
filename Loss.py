import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance


class LossRMSE(nn.Module):
    def __init__(self):
        super(LossRMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse


class SubConLoss(nn.Module):
    def __init__(self, temperature=0.073, scale_by_temperature=False):
        super(SubConLoss, self).__init__()
        self.T = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, feature, label):
        feature = F.normalize(feature, p=2, dim=1)
        batch_size = feature.shape[0]

        # Generate mask
        mask = torch.eq(label, label.T).float()
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).cuda()
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        # Calculate similarity
        anchor_dot_contrast = torch.div(torch.matmul(feature, feature.T), self.T)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        denominator = torch.sum(exp_logits * negatives_mask, dim=1, keepdim=True) + \
                      torch.sum(exp_logits * positives_mask, dim=1, keepdim=True)
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        num_positives_per_row = torch.sum(positives_mask, dim=1)
        log_probs = torch.sum(log_probs * positives_mask, dim=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

        # Loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.T
        loss = loss.mean()

        return loss
