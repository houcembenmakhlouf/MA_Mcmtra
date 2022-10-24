import torch
from torch import nn


class HBMNet(nn.Module):
    def __init__(self, vdim, tdim, nclasses, device):
        super(HBMNet, self).__init__()
        self.device = device

        # tweet
        self.tweet_vfc1 = nn.Linear(vdim, 128)
        self.tweet_vbn1 = nn.BatchNorm1d(128)
        self.tweet_vdp1 = nn.Dropout(0.5)

        self.tweet_tfc1 = nn.Linear(tdim, 128)
        self.tweet_tbn1 = nn.BatchNorm1d(128)
        self.tweet_tdp1 = nn.Dropout(0.5)

        self.tweet_fc2 = nn.Linear(256, 128)
        self.tweet_bn2 = nn.BatchNorm1d(128)
        self.tweet_dp2 = nn.Dropout(0.5)

        # reply
        self.reply_vfc1 = nn.Linear(vdim, 128)
        self.reply_vbn1 = nn.BatchNorm1d(128)
        self.reply_vdp1 = nn.Dropout(0.5)

        self.reply_tfc1 = nn.Linear(tdim, 128)
        self.reply_tbn1 = nn.BatchNorm1d(128)
        self.reply_tdp1 = nn.Dropout(0.5)

        self.reply_fc2 = nn.Linear(256, 128)
        self.reply_bn2 = nn.BatchNorm1d(128)
        self.reply_dp2 = nn.Dropout(0.5)

        self.reply_fc3 = nn.Linear(128, 128)
        self.reply_bn3 = nn.BatchNorm1d(128)
        self.reply_dp3 = nn.Dropout(0.5)

        # quote
        self.quote_vfc1 = nn.Linear(vdim, 128)
        self.quote_vbn1 = nn.BatchNorm1d(128)
        self.quote_vdp1 = nn.Dropout(0.5)

        self.quote_tfc1 = nn.Linear(tdim, 128)
        self.quote_tbn1 = nn.BatchNorm1d(128)
        self.quote_tdp1 = nn.Dropout(0.5)

        self.quote_fc2 = nn.Linear(256, 128)
        self.quote_bn2 = nn.BatchNorm1d(128)
        self.quote_dp2 = nn.Dropout(0.5)

        self.quote_fc3 = nn.Linear(128, 128)
        self.quote_bn3 = nn.BatchNorm1d(128)
        self.quote_dp3 = nn.Dropout(0.5)

        # general
        self.relu = nn.ReLU()
        self.fc = nn.Linear(384, nclasses)

    def forward(self, vfeat, tfeat):
        x1 = self.tweet_vdp1(
            self.relu(self.tweet_vbn1(self.tweet_vfc1(vfeat["tweet"].to(self.device))))
        )
        x2 = self.tweet_tdp1(
            self.relu(self.tweet_tbn1(self.tweet_tfc1(tfeat["tweet"].to(self.device))))
        )
        tweet_x = torch.cat((x1, x2), axis=1)
        tweet_x = self.tweet_dp2(self.relu(self.tweet_bn2(self.tweet_fc2(tweet_x))))

        reply_x = torch.zeros(tfeat["tweet"].shape[0], 128).to(self.device)
        for i, _ in enumerate(tfeat["replies"]):
            x1 = self.reply_vdp1(
                self.relu(
                    self.reply_vbn1(
                        self.reply_vfc1(vfeat["replies"][i].to(self.device))
                    )
                )
            )
            x2 = self.reply_tdp1(
                self.relu(
                    self.reply_tbn1(
                        self.reply_tfc1(tfeat["replies"][i].to(self.device))
                    )
                )
            )
            x3 = torch.cat((x1, x2), axis=1)
            x3 = self.reply_dp2(self.relu(self.reply_bn2(self.reply_fc2(x3))))
            w = tfeat["replies_indices"][:, i].unsqueeze(dim=1).to(self.device)
            x3 = x3 * w
            reply_x += x3

        n_replies = tfeat["replies_indices"].sum(dim=1).to(self.device)
        n_replies[n_replies == 0] = -1

        reply_x = torch.div(torch.transpose(reply_x, 0, 1), n_replies)
        reply_x = torch.transpose(reply_x, 0, 1)

        reply_x = reply_x.nan_to_num()
        reply_x[reply_x == float("Inf")] = 0

        reply_x = self.reply_dp3(self.relu(self.reply_bn3(self.reply_fc3(reply_x))))

        quote_x = torch.zeros(tfeat["tweet"].shape[0], 128).to(self.device)
        for i, _ in enumerate(tfeat["quotes"]):
            x1 = self.quote_vdp1(
                self.relu(
                    self.quote_vbn1(self.quote_vfc1(vfeat["quotes"][i].to(self.device)))
                )
            )
            x2 = self.quote_tdp1(
                self.relu(
                    self.quote_tbn1(self.quote_tfc1(tfeat["quotes"][i].to(self.device)))
                )
            )
            x3 = torch.cat((x1, x2), axis=1)
            x3 = self.quote_dp2(self.relu(self.quote_bn2(self.quote_fc2(x3))))
            w = tfeat["quotes_indices"][:, i].unsqueeze(dim=1).to(self.device)
            x3 = x3 * w
            quote_x += x3

        n_quotes = tfeat["quotes_indices"].sum(dim=1).to(self.device)
        n_quotes[n_quotes == 0] = -1

        quote_x = torch.div(torch.transpose(quote_x, 0, 1), n_quotes)
        quote_x = torch.transpose(quote_x, 0, 1)

        quote_x = quote_x.nan_to_num()
        quote_x[quote_x == float("Inf")] = 0
        quote_x = self.quote_dp3(self.relu(self.quote_bn3(self.quote_fc3(quote_x))))

        x = torch.cat((tweet_x, reply_x, quote_x), axis=1)
        x = self.fc(x)
        return x
