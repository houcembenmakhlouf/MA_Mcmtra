import torch
import logging
import argparse
import random
import shutil
import numpy as np
from torch import nn
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold

from data import HBMDataset
from utils import worker_init_fn, init_experiment
from models import (
    HBMNet,
    HBMNetT,
    HBMNetTV,
    HBMNetAtt,
    HBMNetAttLike,
    HBMNetSelfAtt,
    HBMNetDoubleAtt,
    HBMNetOrepImgt,
)
from train import evaluate, find_last_exp_indx
from utils import plot_classwise_prob, plot_conf_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multimodal MLP Models for Sentiment"
    )
    parser.add_argument(
        "--vtype",
        type=str,
        default="imagenet",
        help="imagenet | places | emotion | clip",
    )
    parser.add_argument(
        "--ttype",
        type=str,
        default="twitterxlmrobertabase",
        help="bertbase | robertabase | clip | xlmrobertabase | twitterxlmrobertabase",
    )
    parser.add_argument("--ftype", type=str, default="feats", help="feats | logits")
    parser.add_argument(
        "--layer", type=str, default="sumavg", help="sumavg, 2last, last"
    )
    parser.add_argument("--smooth", type=bool, default=False, help="False | True")

    parser.add_argument("--gpu", type=int, default=2, help="0,1,..")
    parser.add_argument(
        "--config", type=str, default="configs/config_hbmnet_double_de_test_fr.yaml"
    )
    args = parser.parse_args()
    return args


def get_hbmnet_double_att_embds(model, vfeat, tfeat):
    x1 = model.tweet_vdp1(
        model.relu(model.tweet_vbn1(model.tweet_vfc1(vfeat["tweet"].to(model.device))))
    )
    x2 = model.tweet_tdp1(
        model.relu(model.tweet_tbn1(model.tweet_tfc1(tfeat["tweet"].to(model.device))))
    )
    tweet_x = torch.cat((x1, x2), axis=1)
    tweet_x = model.tweet_dp2(model.relu(model.tweet_bn2(model.tweet_fc2(tweet_x))))

    reply_x_list = []
    for i, _ in enumerate(tfeat["replies"]):
        x1 = model.reply_vdp1(
            model.relu(
                model.reply_vbn1(model.reply_vfc1(vfeat["replies"][i].to(model.device)))
            )
        )
        x2 = model.reply_tdp1(
            model.relu(
                model.reply_tbn1(model.reply_tfc1(tfeat["replies"][i].to(model.device)))
            )
        )
        x3 = torch.cat((x1, x2), axis=1)
        x3 = model.reply_dp2(model.relu(model.reply_bn2(model.reply_fc2(x3))))

        x3, weights = model.reply_self_att(
            query=x3.unsqueeze(dim=1), context=x3.unsqueeze(dim=1)
        )
        x3 = x3.squeeze()

        w = tfeat["replies_indices"][:, i].unsqueeze(dim=1).to(model.device)
        x3 = x3 * w
        reply_x_list.append(x3)

    reply_x = torch.stack(tensors=reply_x_list, dim=1)
    reply_x, weights = model.reply_att(
        query=tweet_x.unsqueeze(dim=1),
        context=reply_x,
        context_mask=tfeat["replies_indices"].to(model.device),
    )
    reply_x = reply_x.squeeze()
    reply_x = model.reply_dp3(model.relu(model.reply_bn3(model.reply_fc3(reply_x))))

    quote_x_list = []
    for i, _ in enumerate(tfeat["quotes"]):
        x1 = model.quote_vdp1(
            model.relu(
                model.quote_vbn1(model.quote_vfc1(vfeat["quotes"][i].to(model.device)))
            )
        )
        x2 = model.quote_tdp1(
            model.relu(
                model.quote_tbn1(model.quote_tfc1(tfeat["quotes"][i].to(model.device)))
            )
        )
        x3 = torch.cat((x1, x2), axis=1)
        x3 = model.quote_dp2(model.relu(model.quote_bn2(model.quote_fc2(x3))))

        x3, weights = model.quote_self_att(
            query=x3.unsqueeze(dim=1), context=x3.unsqueeze(dim=1)
        )
        x3 = x3.squeeze()

        w = tfeat["quotes_indices"][:, i].unsqueeze(dim=1).to(model.device)
        x3 = x3 * w
        quote_x_list.append(x3)

    quote_x = torch.stack(tensors=quote_x_list, dim=1)
    quote_x, weights = model.quote_att(
        query=tweet_x.unsqueeze(dim=1),
        context=quote_x,
        context_mask=tfeat["quotes_indices"].to(model.device),
    )
    quote_x = quote_x.squeeze()
    quote_x = model.quote_dp3(model.relu(model.quote_bn3(model.quote_fc3(quote_x))))

    x = torch.cat((tweet_x, reply_x, quote_x), axis=1)

    return x


def reduce_dimensions(dataloaders):

    with torch.no_grad():
        embds_list = []
        labels_list = []
        if type(dataloaders) != list:
            dataloaders = [dataloaders]

        for dataloader in dataloaders:
            for sample in dataloader:
                embds = get_hbmnet_double_att_embds(
                    model, sample["vfeat"], sample["tfeat"]
                )
                embds_list.append(embds)
                labels_list.append(sample["label"])

    embds = torch.cat(embds_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)
    embds_reduced = tsne.fit_transform(embds.cpu().detach().numpy())

    return embds_reduced, labels


def plot_embds(embds_reduced, label_topics, title, figsize=(12, 10), save_fig=False):

    color_dict = {
        "corona": "black",
        "sport": "gold",
        "politics": "magenta",
    }

    plt.figure(figsize=figsize)
    plt.title(title)
    ax = sns.scatterplot(
        x=embds_reduced[:, 0],
        y=embds_reduced[:, 1],
        hue=label_topics,
        palette=color_dict,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

    plt.show(block=False)
    if save_fig:
        plt.savefig(f"figs/{title}.png")


if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = parse_args()

    img_feat_type = args.ftype
    img_model_type = args.vtype
    txt_model_type = args.ttype
    txt_feat_type = args.layer
    # smooth = args.smooth

    config_path = args.config

    cfg = init_experiment(config_path=config_path)
    exp_name = cfg.exp_name
    model_type = cfg.model

    # saving directory
    exp_parent_path = Path(cfg.training.output_path)
    last_exp_indx = find_last_exp_indx(exp_parent_path)
    exp_idx = f"{last_exp_indx:03d}"
    if exp_name != "":
        exp_name = f"{exp_idx}_{exp_name}"
    else:
        exp_name = f"{exp_idx}"
    exp_path = exp_parent_path / exp_name
    exp_path.mkdir(parents=True)

    # copy the config file
    shutil.copyfile(config_path, str(exp_path / "config.yaml"))

    logging.basicConfig(
        filename=exp_path / "console.log",
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.info("Starting Experiment")

    gpu = args.gpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    train_val_set = HBMDataset(
        data_path=cfg.data.train.data_path,
        txt_feat_path=cfg.data.train.txt_feat_path,
        img_feat_path=cfg.data.train.img_feat_path,
        normalize=cfg.data.normalize,
        txt_model_type=txt_model_type,
        img_model_type=img_model_type,
        txt_feat_type=txt_feat_type,
        img_feat_type=img_feat_type,
        # split="train/val",
        split="full",
        val_percent=cfg.data.val_percent,
        test_percent=cfg.data.test_percent,
        limit_n_replies=cfg.data.limit_n_replies,
        limit_n_quotes=cfg.data.limit_n_quotes,
        use_img_feat=cfg.data.use_img_feat,
        use_like_feat=cfg.data.use_like_feat,
        multilabel=cfg.training.multilabel,
    )
    test_set = HBMDataset(
        data_path=cfg.data.test.data_path,
        txt_feat_path=cfg.data.test.txt_feat_path,
        img_feat_path=cfg.data.test.img_feat_path,
        normalize=cfg.data.normalize,
        txt_model_type=txt_model_type,
        img_model_type=img_model_type,
        txt_feat_type=txt_feat_type,
        img_feat_type=img_feat_type,
        # split="train/val",
        split="full",
        val_percent=cfg.data.val_percent,
        test_percent=cfg.data.test_percent,
        limit_n_replies=cfg.data.limit_n_replies,
        limit_n_quotes=cfg.data.limit_n_quotes,
        use_img_feat=cfg.data.use_img_feat,
        use_like_feat=cfg.data.use_like_feat,
        multilabel=cfg.training.multilabel,
    )

    kfold = StratifiedKFold(
        n_splits=cfg.training.cv.k_folds, shuffle=True, random_state=seed
    )
    folds = [
        next(kfold.split(train_val_set, train_val_set.labels_str))
        for i in range(cfg.training.cv.k_folds)
    ]
    fold = cfg.fold
    train_ids, val_ids = folds[fold]

    # train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)

    valloader = torch.utils.data.DataLoader(
        train_val_set,
        batch_size=cfg.data.batch_size,
        sampler=val_sampler,
        pin_memory=True,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init_fn,
    )
    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.data.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init_fn,
    )

    # loss function
    if cfg.training.multilabel:
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # initialize model
    if model_type == "hbmnet":
        model = HBMNet(
            vdim=test_set.vdim,
            tdim=test_set.tdim,
            nclasses=test_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_t":
        model = HBMNetT(
            # vdim=train_set.vdim,
            tdim=test_set.tdim,
            nclasses=test_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_tv":
        model = HBMNetTV(
            vdim=test_set.vdim,
            tdim=test_set.tdim,
            nclasses=test_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_att":
        model = HBMNetAtt(
            vdim=test_set.vdim,
            tdim=test_set.tdim,
            nclasses=test_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_att_like":
        model = HBMNetAttLike(
            vdim=test_set.vdim,
            tdim=test_set.tdim,
            nclasses=test_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_self_att":
        model = HBMNetSelfAtt(
            vdim=test_set.vdim,
            tdim=test_set.tdim,
            nclasses=test_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_double_att":
        model = HBMNetDoubleAtt(
            vdim=test_set.vdim,
            tdim=test_set.tdim,
            nclasses=test_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_w_rep_imgt":
        model = HBMNetOrepImgt(
            vdim=test_set.vdim,
            tdim=test_set.tdim,
            nclasses=test_set.get_n_classes(),
            device=device,
        )
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(cfg.model_path))
    model.to(device)
    model.eval()

    (
        test_loss,
        test_acc,
        test_f1,
        test_conf_matrix,
        test_classwise_prob,
        _,
        _,
    ) = evaluate(
        model, testloader, device, model_type, criterion, cfg.training.multilabel
    )

    logging.info(f"Test Accuracy: {test_acc} %")
    logging.info(f"Test Loss: {test_loss}")
    logging.info(f"Test F1 Score: {test_f1}")
    logging.info(f"Test Classwise Probability: \n{test_classwise_prob.round(1)}")
    logging.info(f"Test Confusion Matrix: \n{test_conf_matrix.round(1)}")

    plot_classwise_prob(
        probs=test_classwise_prob,
        classes=list(test_set.label_keys.keys()),
        exp_path=exp_path,
        title=f"Test Classwise Probability Fold {fold}",
    )

    plot_conf_matrix(
        conf_matrix=test_conf_matrix,
        classes=list(test_set.label_keys.keys()),
        exp_path=exp_path,
        title=f"Test Confusion Matrix Fold {fold}",
    )

    embds_reduced, labels = reduce_dimensions(dataloaders=[valloader, testloader])

    label_topics = [list(test_set.label_keys.keys())[x] for x in labels]

    plot_embds(
        embds_reduced[: len(val_ids)],
        label_topics[: len(val_ids)],
        title="Embedding_HBMNetDoubleAtt_Valset",
        figsize=(18, 15),
        save_fig=True,
    )

    plot_embds(
        embds_reduced[len(val_ids) :],
        label_topics[len(val_ids) :],
        title="Embedding_HBMNetDoubleAtt_Testset",
        figsize=(18, 15),
        save_fig=True,
    )
