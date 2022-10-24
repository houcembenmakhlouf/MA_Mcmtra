import pickle
import pandas as pd
import numpy as np
import torch
import warnings
import time
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm
from copy import deepcopy


class HBMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        txt_feat_path: str,
        img_feat_path: str,
        normalize: bool,
        txt_model_type: str,
        img_model_type: str,
        txt_feat_type: str,
        img_feat_type: str,
        split: str,
        val_percent: float,
        test_percent: float,
        limit_n_replies: int = None,
        limit_n_quotes: int = None,
        use_img_feat: bool = True,
        use_like_feat: bool = True,
        multilabel: bool = True,
    ):
        self.label_keys = {
            "war": 0,
            "automotiveIndustry": 1,
            "corona": 2,
            "sport": 3,
            "politics": 4,
            "fun": 5,
            "history": 6,
            "economics": 7,
            "promotion": 8,
            "social": 9,
            "weather": 10,
            "technology": 11,
        }
        # for multilingual test with 3 lcasses
        # self.label_keys = {
        #     "corona": 0,
        #     "sport": 1,
        #     "politics": 2,
        # }

        valid_split = {"full", "train", "val", "test", "train/val"}
        if split not in valid_split:
            raise ValueError(f"results: split must be one of {valid_split}.")

        self.normalize = normalize
        self.txt_model_type = txt_model_type
        self.img_model_type = img_model_type
        self.txt_feat_type = txt_feat_type
        self.img_feat_type = img_feat_type

        self.use_img_feat = use_img_feat
        self.use_like_feat = use_like_feat

        self._get_feats_dim()

        # load data
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        # load text features
        with open(txt_feat_path, "rb") as f:
            self.txt_feats = pickle.load(f)

        self._filter_txt_feats()

        self.max_n_replies, self.max_n_quotes = self._get_max_n_replies_n_quotes()

        if limit_n_replies is not None:
            if limit_n_replies > self.max_n_replies:
                warnings.warn(
                    f"limit_n_replies ({limit_n_replies}) is bigger than max_n_replies ({self.max_n_replies})!"
                )
            else:
                self.max_n_replies = limit_n_replies
        if limit_n_quotes is not None:
            if limit_n_quotes > self.max_n_quotes:
                warnings.warn(
                    f"limit_n_quotes ({limit_n_quotes}) is bigger than max_n_quotes ({self.max_n_quotes})!"
                )
            else:
                self.max_n_quotes = limit_n_quotes

        if self.use_img_feat:
            # load img features
            with open(img_feat_path, "rb") as f:
                self.img_feats = pickle.load(f)

            self._filter_img_feats()

        if self.use_like_feat:
            # load like features (already in self.data)
            self.like_feats = []
            self._filter_like_feats()

        # make all replies and quotes in txt and img feats with the same size
        self._prepare_for_batching()

        # normalize data
        if self.normalize:
            self._normalize_feats()

        df = pd.DataFrame(self.data)
        self.labels_topics = df["topic"].tolist()
        self.multilabel = multilabel

        self.labels_str = df["topic"].apply(", ".join).tolist()
        if not multilabel:
            self.multilabel_keys = {
                "automotiveIndustry, economics, technology": "technology",
                "automotiveIndustry, fun": "automotiveIndustry",
                "automotiveIndustry, fun, economics, technology": "technology",
                "fun, promotion": "promotion",
                "politics, economics": "economics",
                "politics, economics, promotion, social": "promotion",
                "politics, promotion, social": "social",
                "war, politics": "war",
                "war, politics, promotion, social": "war",
                "war, automotiveIndustry, fun, economics, technology": "war",
            }
            # for multilingual test with 3 classes
            # self.multilabel_keys = {
            #     "politics, economics": "politics",
            #     "politics, economics, promotion, social": "politics",
            #     "politics, promotion, social": "politics",
            #     "war, politics": "politics",
            # }
            self.labels_str = [
                self.multilabel_keys[x] if "," in x else x for x in self.labels_str
            ]
        # encode labels
        self.labels = []
        self._encode_labels()

        if split != "full":

            idx_train_val, idx_test = train_test_split(
                np.arange(len(self.labels_str)),
                test_size=test_percent,
                stratify=self.labels_str,
                random_state=42,
            )

            train_val_labels_str = [self.labels_str[x] for x in idx_train_val]
            idx_train, idx_val = train_test_split(
                np.arange(len(train_val_labels_str)),
                test_size=val_percent,
                stratify=train_val_labels_str,
                random_state=42,
            )

            if split == "train":
                self.set_indices(idx_train)
            elif split == "val":
                self.set_indices(idx_val)
            elif split == "test":
                self.set_indices(idx_test)
            elif split == "train/val":
                self.set_indices(idx_train_val)
            else:
                raise NotImplementedError

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {"tfeat": self.txt_feats[idx], "label": torch.tensor(self.labels[idx])}

        if self.use_img_feat:
            item["vfeat"] = self.img_feats[idx]
        if self.use_like_feat:
            item["lfeat"] = self.like_feats[idx]

        return item

    def set_indices(self, indices):
        self.data = [self.data[x] for x in indices]
        self.labels = [self.labels[x] for x in indices]
        self.labels_str = [self.labels_str[x] for x in indices]
        self.txt_feats = [self.txt_feats[x] for x in indices]
        if self.use_img_feat:
            self.img_feats = [self.img_feats[x] for x in indices]
        if self.use_like_feat:
            self.like_feats = [self.like_feats[x] for x in indices]

    def get_data(self, idx):
        return self.data[idx]

    def get_topics(self, idx):
        return self.labels_topics[idx]

    def _filter_txt_feats(self):
        # can be slow
        df = pd.DataFrame(self.txt_feats)

        def filter_txt_feats(sample):

            sample_feats = {"tweet": None, "replies": [], "quotes": []}

            sample_feats["tweet"] = torch.FloatTensor(
                sample["tweet"][self.txt_feat_type]
            )

            for reply in sample["replies"]:
                sample_feats["replies"].append(
                    torch.FloatTensor(reply[self.txt_feat_type])
                )

            for quote in sample["quotes"]:
                sample_feats["quotes"].append(
                    torch.FloatTensor(quote[self.txt_feat_type])
                )

            return sample_feats

        tqdm.pandas()
        df["text_feats"] = df.apply(filter_txt_feats, axis=1)
        self.txt_feats = df["text_feats"].tolist()

    def _filter_img_feats(self):
        # can be slow
        df = pd.DataFrame(self.img_feats)

        def filter_img_feats(sample):

            sample_feats = {"tweet": None, "replies": [], "quotes": []}

            if sample["tweet"] is not None:
                sample_feats["tweet"] = torch.FloatTensor(
                    sample["tweet"][self.img_feat_type]
                ).squeeze()
            else:
                sample_feats["tweet"] = torch.FloatTensor(np.zeros((self.vdim)))

            for reply in sample["replies"]:
                if reply is not None:
                    sample_feats["replies"].append(
                        torch.FloatTensor(reply[self.img_feat_type]).squeeze()
                    )
                else:
                    sample_feats["replies"].append(
                        torch.FloatTensor(np.zeros((self.vdim)))
                    )

            for quote in sample["quotes"]:
                if quote is not None:
                    sample_feats["quotes"].append(
                        torch.FloatTensor(quote[self.img_feat_type]).squeeze()
                    )
                else:
                    sample_feats["quotes"].append(
                        torch.FloatTensor(np.zeros((self.vdim)))
                    )

            return sample_feats

        tqdm.pandas()
        df["img_feats"] = df.apply(filter_img_feats, axis=1)
        self.img_feats = df["img_feats"].tolist()

    def _filter_like_feats(self):
        df = pd.DataFrame(self.data)

        def filter_like_feats(sample):

            sample_feats = {"tweet": None, "replies": [], "quotes": []}

            sample_feats["tweet"] = torch.FloatTensor([sample["tweet"]["likesNbr"]])

            for reply in sample["replies"]:
                sample_feats["replies"].append(torch.FloatTensor([reply["likesNbr"]]))

            for quote in sample["quotes"]:
                sample_feats["quotes"].append(torch.FloatTensor([quote["likesNbr"]]))

            return sample_feats

        tqdm.pandas()
        df["like_feats"] = df.apply(filter_like_feats, axis=1)
        self.like_feats = df["like_feats"].tolist()

    def _get_feats_dim(self):
        if self.txt_model_type in ["xlmrobertabase", "twitterxlmrobertabase"]:
            tdim = 3072 if "catavg" in self.txt_feat_type else 768
        else:
            raise NotImplementedError

        if self.img_model_type == "imagenet":
            vdim = 2048 if self.img_feat_type == "feats" else 1000
        else:
            raise NotImplementedError

        self.tdim, self.vdim = tdim, vdim

    def get_n_classes(self):
        return len(self.label_keys)

    def get_unique_class_combs(self):
        df = pd.DataFrame(self.data)
        return df["topic"].apply(", ".join).unique().tolist()

    def _get_max_n_replies_n_quotes(self):
        df = pd.DataFrame(self.txt_feats)

        df["n_replies"] = df.apply(lambda x: len(x["replies"]), axis=1)
        df["n_quotes"] = df.apply(lambda x: len(x["quotes"]), axis=1)

        max_n_replies = df["n_replies"].max()
        max_n_quotes = df["n_quotes"].max()

        return max_n_replies, max_n_quotes

    def _prepare_for_batching(self):
        def prepare_txt_feats(sample):

            sample_feats = {
                "tweet": None,
                "replies": [],
                "replies_indices": [],
                "quotes": [],
                "quotes_indices": [],
            }

            sample_feats["tweet"] = sample["tweet"]

            sample["replies"] = sample["replies"][: self.max_n_replies]
            sample_feats["replies"] = sample["replies"] + [torch.zeros((self.tdim))] * (
                self.max_n_replies - len(sample["replies"])
            )
            sample_feats["replies_indices"] = torch.tensor(
                [1] * len(sample["replies"])
                + [0] * (self.max_n_replies - len(sample["replies"]))
            )

            sample["quotes"] = sample["quotes"][: self.max_n_quotes]
            sample_feats["quotes"] = sample["quotes"] + [torch.zeros((self.tdim))] * (
                self.max_n_quotes - len(sample["quotes"])
            )
            sample_feats["quotes_indices"] = torch.tensor(
                [1] * len(sample["quotes"])
                + [0] * (self.max_n_quotes - len(sample["quotes"]))
            )

            return sample_feats

        def prepare_img_feats(sample):

            sample_feats = {
                "tweet": None,
                "replies": [],
                "replies_indices": [],
                "quotes": [],
                "quotes_indices": [],
            }

            sample_feats["tweet"] = sample["tweet"]

            sample["replies"] = sample["replies"][: self.max_n_replies]
            sample_feats["replies"] = sample["replies"] + [torch.zeros((self.vdim))] * (
                self.max_n_replies - len(sample["replies"])
            )
            sample_feats["replies_indices"] = torch.tensor(
                [1] * len(sample["replies"])
                + [0] * (self.max_n_replies - len(sample["replies"]))
            )

            sample["quotes"] = sample["quotes"][: self.max_n_quotes]
            sample_feats["quotes"] = sample["quotes"] + [torch.zeros((self.vdim))] * (
                self.max_n_quotes - len(sample["quotes"])
            )
            sample_feats["quotes_indices"] = torch.tensor(
                [1] * len(sample["quotes"])
                + [0] * (self.max_n_quotes - len(sample["quotes"]))
            )

            return sample_feats

        def prepare_like_feats(sample):

            sample_feats = {
                "tweet": None,
                "replies": [],
                "replies_indices": [],
                "quotes": [],
                "quotes_indices": [],
            }

            sample_feats["tweet"] = sample["tweet"]

            sample["replies"] = sample["replies"][: self.max_n_replies]
            sample_feats["replies"] = sample["replies"] + [torch.FloatTensor([0])] * (
                self.max_n_replies - len(sample["replies"])
            )
            sample_feats["replies_indices"] = torch.tensor(
                [1] * len(sample["replies"])
                + [0] * (self.max_n_replies - len(sample["replies"]))
            )

            sample["quotes"] = sample["quotes"][: self.max_n_quotes]
            sample_feats["quotes"] = sample["quotes"] + [torch.FloatTensor([0])] * (
                self.max_n_quotes - len(sample["quotes"])
            )
            sample_feats["quotes_indices"] = torch.tensor(
                [1] * len(sample["quotes"])
                + [0] * (self.max_n_quotes - len(sample["quotes"]))
            )

            return sample_feats

        tqdm.pandas()

        df = pd.DataFrame(self.txt_feats)
        df["text_feats"] = df.apply(prepare_txt_feats, axis=1)
        self.txt_feats = df["text_feats"].tolist()

        if self.use_img_feat:
            df = pd.DataFrame(self.img_feats)
            df["img_feats"] = df.apply(prepare_img_feats, axis=1)
            self.img_feats = df["img_feats"].tolist()
        if self.use_like_feat:
            df = pd.DataFrame(self.like_feats)
            df["like_feats"] = df.apply(prepare_like_feats, axis=1)
            self.like_feats = df["like_feats"].tolist()

    def _normalize_feats(self):
        print("Starting Normalization")
        start = time.time()

        def normalize_tv_feat(feat):
            # normalize feats
            # feat_normalized = deepcopy(feat)
            feat_normalized = {
                "tweet": None,
                "replies": feat["replies"],
                "replies_indices": feat["replies_indices"],
                "quotes": feat["quotes"],
                "quotes_indices": feat["quotes_indices"],
            }
            feat_normalized["tweet"] = torch.FloatTensor(
                preprocessing.normalize(feat["tweet"].reshape(1, -1), axis=1).flatten()
            )
            for i, reply in enumerate(feat["replies"]):
                if reply.sum() != 0:
                    feat_normalized["replies"][i] = torch.FloatTensor(
                        preprocessing.normalize(reply.reshape(1, -1), axis=1).flatten()
                    )
            for i, quote in enumerate(feat["quotes"]):
                if quote.sum() != 0:
                    feat_normalized["quotes"][i] = torch.FloatTensor(
                        preprocessing.normalize(quote.reshape(1, -1), axis=1).flatten()
                    )

            return feat_normalized

        def normalize_l_feat(feat):
            feat_normalized = {
                "tweet": feat["tweet"],
                "replies": feat["replies"],
                "replies_indices": feat["replies_indices"],
                "quotes": feat["quotes"],
                "quotes_indices": feat["quotes_indices"],
            }

            max_likes_replies = max(feat["replies"])
            if max_likes_replies != 0:
                feat_normalized["replies"] = [
                    torch.FloatTensor(x / max_likes_replies) for x in feat["replies"]
                ]
            max_likes_quotes = max(feat["quotes"])
            if max_likes_quotes != 0:
                feat_normalized["quotes"] = [
                    torch.FloatTensor(x / max_likes_quotes) for x in feat["quotes"]
                ]

            return feat_normalized

        df = pd.DataFrame(self.txt_feats)
        df["txt_feats"] = df.apply(normalize_tv_feat, axis=1)
        self.txt_feats = df["txt_feats"].tolist()

        if self.use_img_feat:
            df = pd.DataFrame(self.img_feats)
            df["img_feats"] = df.apply(normalize_tv_feat, axis=1)
            self.img_feats = df["img_feats"].tolist()

        if self.use_like_feat:
            df = pd.DataFrame(self.like_feats)
            df["like_feats"] = df.apply(normalize_l_feat, axis=1)
            self.like_feats = df["like_feats"].tolist()

        end = time.time()
        print(f"Normalization is finished in {round(end - start, 2)}s.")

    def _encode_labels(self):
        if self.multilabel:
            self.labels = []
            for label_topics in self.labels_topics:
                label = np.zeros((self.get_n_classes()))
                for x in label_topics:
                    label[self.label_keys[x]] = 1

                self.labels.append(label)
        else:
            self.labels = []
            for label_str in self.labels_str:
                label = self.label_keys[label_str]
                self.labels.append(label)
