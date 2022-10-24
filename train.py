import torch
import logging
import sqlite3
import random, copy
import time
import json
import pickle
import re
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from data import HBMDataset
from utils import worker_init_fn, init_experiment
from models import (
    HBMNet,
    HBMNetT,
    HBMNetTV,
    HBMNetAtt,
)
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
        default="xlmrobertabase",
        help="bertbase | robertabase | clip | xlmrobertabase",
    )
    parser.add_argument("--ftype", type=str, default="feats", help="feats | logits")
    parser.add_argument(
        "--layer", type=str, default="sumavg", help="sumavg, 2last, last"
    )
    parser.add_argument("--smooth", type=bool, default=False, help="False | True")

    parser.add_argument("--gpu", type=int, default=3, help="0,1,..")
    parser.add_argument(
        "--model",
        type=str,
        default="hbmnet_att_like",
        help="hbmnet | hbmnet_t | hbmnet_tv | hbmnet_att | hbmnet_att_like | hbmnet_w_rep_imgt",
    )
    args = parser.parse_args()
    return args


def train(
    model,
    model_type,
    img_model_type,
    img_feat_type,
    txt_model_type,
    txt_feat_type,
    optimizer,
    device,
    criterion,
    lr_scheduler,
    num_epochs,
    trainloader,
    valloader,
    output_path,
    save_every,
    patience,
    multilabel,
):
    since = time.time()

    # Set up Tensorboard
    writer = SummaryWriter(log_dir=output_path / "logs")

    best_model = model
    best_acc = 0.0
    best_val_loss = 100
    best_val_f1 = 0.0
    best_epoch = 0
    current_patience = 0

    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}/{num_epochs}")
        logging.info("-" * 10)

        since2 = time.time()

        model.train()  # Set model to training mode

        running_loss = 0.0

        cnt = 0
        y_true = []
        y_pred = []
        # Iterate over data.
        for batch in trainloader:

            labels = batch["label"]
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            if model_type in [
                "hbmnet",
                "hbmnet_tv",
                "hbmnet_att",
                "hbmnet_self_att",
                "hbmnet_double_att",
                "hbmnet_w_rep_imgt",
            ]:
                outputs = model(batch["vfeat"], batch["tfeat"])
            elif model_type == "hbmnet_t":
                outputs = model(batch["tfeat"])
            elif model_type == "hbmnet_att_like":
                outputs = model(batch["vfeat"], batch["tfeat"], batch["lfeat"])
            else:
                raise NotImplementedError

            y_true = y_true + labels.tolist()

            if multilabel:
                preds = torch.round(torch.sigmoid(outputs))
            else:
                preds = torch.argmax(outputs, dim=1)

            y_pred = y_pred + preds.tolist()

            loss = criterion(outputs, labels)
            # another loss function method - see helpers for implementation
            # loss = cal_loss(outputs, labels, smoothing=smooth)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()

            if cnt % 1 == 0:
                logging.info(
                    "[%d, %5d] loss: %.5f, Acc: %.2f"
                    % (
                        epoch,
                        cnt + 1,
                        loss.item(),
                        100.0 * accuracy_score(y_true, y_pred),
                    )
                )

            cnt = cnt + 1

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * accuracy_score(y_true, y_pred)

        logging.info("Training Loss: {:.6f} Acc: {:.2f}".format(train_loss, train_acc))

        val_loss, val_acc, val_f1, val_conf_matrix, val_classwise_prob, _, _ = evaluate(
            model, valloader, device, model_type, criterion, multilabel
        )

        logging.info(
            "Epoch: {:d}, Val Loss: {:.4f}, Val Acc: {:.4f}%, Val F1: {:.4f}, \nConfusion Matrix: \n{}, \nClasswise Probabilities: \n{}".format(
                epoch,
                val_loss,
                val_acc,
                val_f1,
                val_conf_matrix.round(1),
                val_classwise_prob.round(1),
            )
        )

        # compute classwise accuracy

        # write to Tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)

        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        writer.add_scalar("F1/val", val_f1, epoch)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group["lr"]

        writer.add_scalar("lr", get_lr(optimizer), epoch)

        fig = plot_classwise_prob(
            probs=val_classwise_prob,
            classes=list(valloader.dataset.label_keys.keys()),
            title=f"Classwise Probability: Epoch {epoch}",
        )
        writer.add_figure("Classwise Probability", fig, epoch)
        fig = plot_conf_matrix(
            conf_matrix=val_conf_matrix,
            classes=list(valloader.dataset.label_keys.keys()),
            title=f"Confusion Matrix: Epoch {epoch}",
        )
        writer.add_figure("Confusion Matrix", fig, epoch)

        if lr_scheduler:
            lr_scheduler.step(val_loss)

        # deep copy the model
        if val_acc >= best_acc:
            best_acc = val_acc
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_conf_matrix = val_conf_matrix
            best_classwise_prob = val_classwise_prob
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            current_patience = 0
        else:
            current_patience += 1

        if epoch % save_every == 0:
            torch.save(
                best_model.state_dict(),
                output_path
                / "checkpoints"
                / f"hbmnet_{img_model_type}_{img_feat_type}_{txt_model_type}_{txt_feat_type}_{epoch}.pt",
            )

        time_elapsed2 = time.time() - since2
        logging.info(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed2 // 60, time_elapsed2 % 60
            )
        )

        if current_patience > patience:
            break

    writer.flush()
    writer.close()

    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    torch.save(
        best_model.state_dict(),
        output_path
        / "checkpoints"
        / f"hbmnet_{img_model_type}_{img_feat_type}_{txt_model_type}_{txt_feat_type}_final.pt",
    )

    logging.info(
        "Best Epoch: {:d}, Best Val Loss: {:.4f}, Best Val Acc: {:.4f}%, Best Val F1: {:.4f}, \nBest Confusion Matrix: \n{}, \nBest Classwise Probabilities: \n{}".format(
            best_epoch,
            best_val_loss,
            best_acc,
            best_val_f1,
            best_conf_matrix.round(1),
            best_classwise_prob.round(1),
        )
    )

    best_results = {
        "best_epoch": best_epoch,
        "best_acc_%": best_acc,
        "best_loss": best_val_loss,
        "best_f1": best_val_f1,
        "best_confusion_matrix": best_conf_matrix.tolist(),
        "best_classwise_prob_%": best_classwise_prob.tolist(),
    }

    return best_results


def evaluate(model, loader, device, model_type, criterion, multilabel):
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    classwise_prob = torch.zeros(
        loader.dataset.get_n_classes(), loader.dataset.get_n_classes(), device=device
    )  # each row contains the accumulated probabilities of each class
    total = torch.zeros(loader.dataset.get_n_classes(), device=device)

    with torch.no_grad():
        for batch in loader:

            labels = batch["label"]
            labels = labels.to(device)

            if model_type in [
                "hbmnet",
                "hbmnet_tv",
                "hbmnet_att",
                "hbmnet_self_att",
                "hbmnet_double_att",
                "hbmnet_w_rep_imgt",
            ]:
                outputs = model(batch["vfeat"], batch["tfeat"])
            elif model_type == "hbmnet_t":
                outputs = model(batch["tfeat"])
            elif model_type == "hbmnet_att_like":
                outputs = model(batch["vfeat"], batch["tfeat"], batch["lfeat"])
            else:
                raise NotImplementedError

            if multilabel:
                probs = torch.sigmoid(outputs)
                preds = torch.round(probs)
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

            test_loss += criterion(outputs, labels).item()
            # test_loss += cal_loss(outputs, labels, smoothing=smooth).item()

            y_true = y_true + labels.tolist()
            y_pred = y_pred + preds.tolist()

            for i in range(probs.shape[0]):
                classwise_prob[labels[i], :] += probs[i, :]
                total[labels[i]] += 1

        acc = 100.0 * accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        conf_matrix = confusion_matrix(y_true, y_pred, normalize="true") * 100

        for i in range(classwise_prob.shape[0]):
            if total[i] != 0:
                classwise_prob[i, :] = classwise_prob[i, :] / total[i]

        classwise_prob = (classwise_prob * 100).cpu().detach().numpy()

    return test_loss / len(loader), acc, f1, conf_matrix, classwise_prob, y_pred, y_true


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

    config_path = "config.yaml"
    cfg = init_experiment(config_path=config_path)

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
    smooth = args.smooth
    model_type = args.model

    gpu = args.gpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    train_set = HBMDataset(
        data_path=cfg.data.data_path,
        normalize=cfg.data.normalize,
        txt_model_type=txt_model_type,
        img_model_type=img_model_type,
        txt_feat_type=txt_feat_type,
        img_feat_type=img_feat_type,
        split="train",
        val_percent=cfg.data.val_percent,
        test_percent=cfg.data.test_percent,
        limit_n_replies=cfg.data.limit_n_replies,
        limit_n_quotes=cfg.data.limit_n_quotes,
    )
    val_set = HBMDataset(
        data_path=cfg.data.data_path,
        normalize=cfg.data.normalize,
        txt_model_type=txt_model_type,
        img_model_type=img_model_type,
        txt_feat_type=txt_feat_type,
        img_feat_type=img_feat_type,
        split="val",
        val_percent=cfg.data.val_percent,
        test_percent=cfg.data.test_percent,
        limit_n_replies=cfg.data.limit_n_replies,
        limit_n_quotes=cfg.data.limit_n_quotes,
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

    trainloader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=worker_init_fn,
    )
    # testloader = DataLoader(
    #     test_set,
    #     batch_size=cfg.data.batch_size,
    #     shuffle=False,
    #     num_workers=cfg.data.num_workers,
    # )

    criterion = nn.BCEWithLogitsLoss().to(device)

    if model_type == "hbmnet":
        model_ft = HBMNet(
            vdim=train_set.vdim,
            tdim=train_set.tdim,
            nclasses=train_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_t":
        model_ft = HBMNetT(
            # vdim=train_set.vdim,
            tdim=train_set.tdim,
            nclasses=train_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_tv":
        model_ft = HBMNetTV(
            vdim=train_set.vdim,
            tdim=train_set.tdim,
            nclasses=train_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_att":
        model_ft = HBMNetAtt(
            vdim=train_set.vdim,
            tdim=train_set.tdim,
            nclasses=train_set.get_n_classes(),
            device=device,
        )
    elif model_type == "hbmnet_self_att":
        model_ft = HBMNetAtt(
            vdim=train_set.vdim,
            tdim=train_set.tdim,
            nclasses=train_set.get_n_classes(),
            device=device,
        )

    model_ft.to(device)
    logging.info(model_ft)

    optimizer_ft = optim.Adam(model_ft.parameters(), cfg.training.init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft,
        mode="min",
        patience=cfg.training.lr_patience,
        verbose=True,
        factor=0.1,
    )

    exp_parent_path = Path(cfg.training.output_path)
    last_exp_indx = find_last_exp_indx(exp_parent_path)
    exp_name = f"{last_exp_indx:03d}"
    exp_path = exp_parent_path / exp_name
    exp_path.mkdir(parents=True)
    (exp_path / "checkpoints").mkdir(parents=True)
    (exp_path / "logs").mkdir(parents=True)

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
        output_path=exp_path,
        save_every=cfg.training.save_every,
        patience=cfg.training.patience,
    )
