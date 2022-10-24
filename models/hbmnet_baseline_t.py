import torch
from torch import nn


class HBMNetT(nn.Module):
    def __init__(self, tdim, nclasses, device):
        super(HBMNetT, self).__init__()
        self.device = device

        # tweet
        self.tweet_tfc1 = nn.Linear(tdim, 128)
        self.tweet_tbn1 = nn.BatchNorm1d(128)
        self.tweet_tdp1 = nn.Dropout(0.5)

        # general
        self.relu = nn.ReLU()
        self.cf = nn.Linear(128, nclasses)

    def forward(self, tfeat):
        x = self.tweet_tdp1(
            self.relu(self.tweet_tbn1(self.tweet_tfc1(tfeat["tweet"].to(self.device))))
        )

        x = self.cf(x)
        return x
