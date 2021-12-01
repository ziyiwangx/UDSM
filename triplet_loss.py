import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import itertools
import pdb
import math


class TripletLoss(nn.Module):
    def __init__(self, margin, device):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, h_c1, h_c2, h_c3, triplets):
        triplets = np.array(triplets)
        sq_ap = (h_c1[triplets[:, 0]] - h_c2[triplets[:, 1]]).pow(2).sum(1)
        sq_an = (h_c1[triplets[:, 0]] - h_c3[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(self.margin + sq_ap - sq_an)

        # triplet_count = losses.size()[0] / h_c1.size()[0]
        triplet_loss = losses.mean() + 1e-16

        return triplet_loss

