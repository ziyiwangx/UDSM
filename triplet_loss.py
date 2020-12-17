import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import itertools
import pdb
import math

class TripletLoss(nn.Module):
    #pass
    def __init__(self,margin,device):
        super(TripletLoss, self).__init__()
        self.margin=margin
        self.device=device

    def similarity(self,label1,label2):
        return label1==label2 # default with singe label

    def forward(self,h_c1, h_c2, h_c3, triplets):
        triplets = np.array(triplets)
        #np.random.shuffle(triplets)
        #print triplets
        #h_c1 = h_c1.cpu().data.numpy()
        #h_c2 = h_c2.cpu().data.numpy()
        #h_c3 = h_c3.cpu().data.numpy()
        #for i in range(len(triplets)):
        #   sq_ap += (np.array(h_c1)[triplets[i, 0]]-np.array(h_c2)[triplets[i, 1]]).pow(2).sum(1)

        #pdb.set_trace()
        #sq_ap = torch.rsqrt((h_c1[triplets[:, 0]] - h_c2[triplets[:, 1]]).pow(2).sum(1))
        #sq_an = torch.rsqrt((h_c1[triplets[:, 0]] - h_c3[triplets[:, 2]]).pow(2).sum(1))
        sq_ap = (h_c1[triplets[:, 0]] - h_c2[triplets[:, 1]]).pow(2).sum(1)
        sq_an = (h_c1[triplets[:, 0]] - h_c3[triplets[:, 2]]).pow(2).sum(1)
        #print(np.array(h_c1).shape, sq_ap.shape)
        losses = F.relu(self.margin + sq_ap - sq_an)

        '''mse_criterion1 = nn.MSELoss()
        mse_criterion1.cuda()
        sq_ap1 = mse_criterion1(np.array(h_c1)[triplets[:, 0]], np.array(h_c2)[triplets[:, 1]])
        sq_ap1 = mse_criterion1(h_c1[triplets[:, 0]], h_c2[triplets[:, 1]])
        sq_an1 = mse_criterion1(h_c1[triplets[:, 0]], h_c3[triplets[:, 2]])
        sq_an1 = mse_criterion1(np.array(h_c1)[triplets[:, 0]], np.array(h_c3)[triplets[:, 2]])
        losses = F.relu(torch.from_numpy(self.margin + sq_ap1 - sq_an1).float().to(self.device))'''

        triplet_count = losses.size()[0] / h_c1.size()[0]
        #pdb.set_trace()
        # semihard_triplet_count=(losses!=0).sum().float().to(self.device)
        #triplet_loss = losses.sum()/triplet_count + 1e-6
        triplet_loss = losses.mean() + 1e-16

        # if semihard_triplet_count>0:
        #     semihard_triplet_loss=losses.sum()/semihard_triplet_count
        # print ("triplet_count", triplet_count)
        # print ("semihard_triplet_count", semihard_triplet_count)
        # print ("triplet_loss:",triplet_loss.item())
        # print ("semihard_triplet_loss",semihard_triplet_loss.item())
        
        return triplet_loss
        # return semihard_triplet_loss


