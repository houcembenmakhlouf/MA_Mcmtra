import torch
from torch import nn


class HBMNetTV(nn.Module):
    def __init__(self, vdim, tdim, nclasses, device):
        super(HBMNetTV, self).__init__()
        self.device = device

        # tweet
        self.tweet_vfc1 = nn.Linear(vdim, 128)
        self.tweet_tfc1 = nn.Linear(tdim, 128)
        self.tweet_vbn1 = nn.BatchNorm1d(128)
        self.tweet_tbn1 = nn.BatchNorm1d(128)
        self.tweet_vdp1 = nn.Dropout(0.8)
        self.tweet_tdp1 = nn.Dropout(0.8)

        # general
        self.relu = nn.ReLU()
        self.cf = nn.Linear(256, nclasses)

    def forward(self, vfeat, tfeat):
        x1 = self.tweet_vdp1(
            self.relu(self.tweet_vbn1(self.tweet_vfc1(vfeat["tweet"].to(self.device))))
        )
        x2 = self.tweet_tdp1(
            self.relu(self.tweet_tbn1(self.tweet_tfc1(tfeat["tweet"].to(self.device))))
        )
        tweet_x = torch.cat((x1, x2), axis=1)

        x = self.cf(tweet_x)
        return x
