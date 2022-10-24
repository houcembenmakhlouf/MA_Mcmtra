import torch


def load_model(dloc):
    model = torch.load(dloc)
    # model.eval()
    print(model)


if __name__ == "__main__":
    # dloc = "exps/027_hbmnet/4/checkpoints/hbmnet_imagenet_feats_twitterxlmrobertabase_sumavg_final.pt"
    dloc = "exps/030_hbmnet_att/4/checkpoints/hbmnet_imagenet_feats_twitterxlmrobertabase_sumavg_final.pt"
    load_model(dloc)
