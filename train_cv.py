import torch
import logging
import argparse
import random
import json
import shutil
import numpy as np
import torch.optim as optim
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from pathlib import Path
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
from train import train
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
        "--config", type=str, default="configs/config_hbmnet_baseline_tv.yaml"
    )
    args = parser.parse_args()
    return args


def find_last_exp_indx(exp_parent_path):
    indx = 0
    for x in exp_parent_path.iterdir():
        if not x.is_dir():
            continue

        exp_name = x.stem
        if exp_name[:3].isnumeric():
            indx = max(indx, int(exp_name[:3]) + 1)

    return indx


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

    # dloc = "data/db_hbm/"
    # samples = prepare_dataset(dloc)
    train_val_set = HBMDataset(
        data_path=cfg.data.data_path,
        txt_feat_path=cfg.data.txt_feat_path,
        img_feat_path=cfg.data.img_feat_path,
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

    # test_set = HBMDataset(
    #     data_path=cfg.data.data_path,
    #     normalize=True,
    #     txt_model_type=txt_model_type,
    #     img_model_type=img_model_type,
    #     txt_feat_type=txt_feat_type,
    #     img_feat_type=img_feat_type,
    #     split="test",
    #     val_percent=cfg.data.val_percent,
    #     test_percent=cfg.data.test_percent,
    #     limit_n_replies=cfg.data.limit_n_replies,
    #     limit_n_quotes=cfg.data.limit_n_quotes,
    # )

    # loss function
    if cfg.training.multilabel:
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Define the K-fold Cross Validator
    if cfg.training.cv.stratified:
        kfold = StratifiedKFold(
            n_splits=cfg.training.cv.k_folds, shuffle=True, random_state=seed
        )
    else:
        kfold = KFold(n_splits=cfg.training.cv.k_folds, shuffle=True, random_state=seed)

    # For fold results
    results = {}

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(
        kfold.split(train_val_set, train_val_set.labels_str)
    ):
        # Print
        logging.info(f"FOLD {fold}")
        logging.info("--------------------------------")

        # class weights
        weight = {}
        for t in np.unique(train_val_set.labels_str):
            weight[t] = 1 / (
                np.array(train_val_set.labels_str)[train_ids] == t
            ).sum().astype(float)

        samples_weight = torch.tensor(
            [
                weight[t] if idx in train_ids else 0
                for idx, t in enumerate(train_val_set.labels_str)
            ]
        )

        # Sample elements randomly from a given list of ids, no replacement.
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, num_samples=len(train_ids)
        )
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            train_val_set,
            batch_size=cfg.data.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        valloader = torch.utils.data.DataLoader(
            train_val_set,
            batch_size=cfg.data.batch_size,
            sampler=val_sampler,
            pin_memory=True,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            worker_init_fn=worker_init_fn,
        )

        # initialize model
        if model_type == "hbmnet":
            model_ft = HBMNet(
                vdim=train_val_set.vdim,
                tdim=train_val_set.tdim,
                nclasses=train_val_set.get_n_classes(),
                device=device,
            )
        elif model_type == "hbmnet_t":
            model_ft = HBMNetT(
                # vdim=train_set.vdim,
                tdim=train_val_set.tdim,
                nclasses=train_val_set.get_n_classes(),
                device=device,
            )
        elif model_type == "hbmnet_tv":
            model_ft = HBMNetTV(
                vdim=train_val_set.vdim,
                tdim=train_val_set.tdim,
                nclasses=train_val_set.get_n_classes(),
                device=device,
            )
        elif model_type == "hbmnet_att":
            model_ft = HBMNetAtt(
                vdim=train_val_set.vdim,
                tdim=train_val_set.tdim,
                nclasses=train_val_set.get_n_classes(),
                device=device,
            )
        elif model_type == "hbmnet_att_like":
            model_ft = HBMNetAttLike(
                vdim=train_val_set.vdim,
                tdim=train_val_set.tdim,
                nclasses=train_val_set.get_n_classes(),
                device=device,
            )
        elif model_type == "hbmnet_self_att":
            model_ft = HBMNetSelfAtt(
                vdim=train_val_set.vdim,
                tdim=train_val_set.tdim,
                nclasses=train_val_set.get_n_classes(),
                device=device,
            )
        elif model_type == "hbmnet_double_att":
            model_ft = HBMNetDoubleAtt(
                vdim=train_val_set.vdim,
                tdim=train_val_set.tdim,
                nclasses=train_val_set.get_n_classes(),
                device=device,
            )
        elif model_type == "hbmnet_w_rep_imgt":
            model_ft = HBMNetOrepImgt(
                vdim=train_val_set.vdim,
                tdim=train_val_set.tdim,
                nclasses=train_val_set.get_n_classes(),
                device=device,
            )
        else:
            raise NotImplementedError

        model_ft.to(device)
        logging.info(model_ft)

        # Initialize optimizer
        optimizer_ft = optim.Adam(
            model_ft.parameters(), cfg.training.init_lr, weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ft,
            mode="min",
            patience=cfg.training.lr_patience,
            verbose=True,
            factor=0.1,
        )

        # fold saving directory
        fold_path = exp_path / str(fold)
        fold_path.mkdir(parents=True)
        (fold_path / "checkpoints").mkdir(parents=True)
        (fold_path / "logs").mkdir(parents=True)

        # train fold
        best_results = train(
            model=model_ft,
            model_type=model_type,
            img_model_type=img_model_type,
            img_feat_type=img_feat_type,
            txt_model_type=txt_model_type,
            txt_feat_type=txt_feat_type,
            optimizer=optimizer_ft,
            device=device,
            criterion=criterion,
            lr_scheduler=scheduler,
            num_epochs=cfg.training.epochs,
            trainloader=trainloader,
            valloader=valloader,
            output_path=fold_path,
            save_every=cfg.training.save_every,
            patience=cfg.training.patience,
            multilabel=cfg.training.multilabel,
        )

        results[fold] = best_results

        with open(exp_path / "cross_val_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        plot_classwise_prob(
            probs=best_results["best_classwise_prob_%"],
            classes=list(train_val_set.label_keys.keys()),
            exp_path=exp_path,
            title=f"Classwise Probability Fold {fold}",
        )

        plot_conf_matrix(
            conf_matrix=best_results["best_confusion_matrix"],
            classes=list(train_val_set.label_keys.keys()),
            exp_path=exp_path,
            title=f"Confusion Matrix Fold {fold}",
        )

    # Print fold results
    logging.info(f"K-FOLD CROSS VALIDATION RESULTS FOR {cfg.training.cv.k_folds} FOLDS")
    logging.info("--------------------------------")
    sum_acc = 0.0
    sum_loss = 0.0
    sum_f1 = 0.0
    sum_classwise_prob = None
    sum_conf_matrix = None
    for key, value in results.items():
        sum_acc += value["best_acc_%"]
        sum_loss += value["best_loss"]
        sum_f1 += value["best_f1"]
        if sum_conf_matrix is None:
            sum_conf_matrix = np.array(value["best_confusion_matrix"])
        else:
            sum_conf_matrix += np.array(value["best_confusion_matrix"])
        if sum_classwise_prob is None:
            sum_classwise_prob = np.array(value["best_classwise_prob_%"])
        else:
            sum_classwise_prob += np.array(value["best_classwise_prob_%"])

    avg_acc = sum_acc / len(results.items())
    avg_loss = sum_loss / len(results.items())
    avg_f1 = sum_f1 / len(results.items())
    avg_classwise_prob = sum_classwise_prob / len(results.items())
    avg_conf_matrix = sum_conf_matrix / len(results.items())

    logging.info(f"Average Accuracy: {avg_acc} %")
    logging.info(f"Average Loss: {avg_loss}")
    logging.info(f"Average F1 Score: {avg_f1}")
    logging.info(f"Average Classwise Probability: \n{avg_classwise_prob.round(1)}")
    logging.info(f"Average Confusion Matrix: \n{avg_conf_matrix.round(1)}")

    results["avg"] = {
        "best_acc_%": avg_acc,
        "best_loss": avg_loss,
        "best_f1": avg_f1,
        "best_classwise_prob_%": avg_classwise_prob.tolist(),
        "best_confusion_matrix": avg_conf_matrix.tolist(),
    }

    with open(exp_path / "cross_val_results.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    plot_classwise_prob(
        probs=avg_classwise_prob,
        classes=list(train_val_set.label_keys.keys()),
        exp_path=exp_path,
        title=f"Classwise Probability Average",
    )

    plot_conf_matrix(
        conf_matrix=avg_conf_matrix,
        classes=list(train_val_set.label_keys.keys()),
        exp_path=exp_path,
        title=f"Confusion Matrix Average",
    )
