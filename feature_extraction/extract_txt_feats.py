import json
import string
import pickle
import torch
import pickle
import numpy as np
from matplotlib.pyplot import text
from tqdm import tqdm

from transformers import (
    BertTokenizer,
    BertModel,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    AutoTokenizer,
    AutoModel,
)

from helpers import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract BERT Features")
    parser.add_argument(
        "--btype",
        type=str,
        default="twitterxlmrobertabase",
        help="bertbase | robertabase | xlmrobertabase | twitterxlmrobertabase | twitterxlmrobertabasesentiment",
    )
    parser.add_argument("--gpu", type=int, default=2, help="0,1,..")
    parser.add_argument("--mvsa", type=str, default="single", help="single | multiple")
    parser.add_argument("--ht", type=bool, default=True, help="True | False")

    args = parser.parse_args()
    return args


def get_text_embed_dict(txt_inps, model, tokenizer, device):

    (
        sent_word_catavg,
        sent_word_sumavg,
        sent_emb_2_last,
        sent_emb_last,
    ) = get_bert_embeddings(txt_inps, model, tokenizer, device)

    embed_dict = {
        "catavg": sent_word_catavg,
        "sumavg": sent_word_sumavg,
        "2last": sent_emb_2_last,
        "last": sent_emb_last,
    }

    return embed_dict


def extract_txt_feats(args):

    gpu = args.gpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    bert_type = {
        "bertbase": (BertModel, BertTokenizer, "bert-base-uncased"),
        "robertabase": (RobertaModel, RobertaTokenizer, "roberta-base"),
        "xlmrobertabase": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-base"),
        "twitterxlmrobertabase": (
            AutoModel,
            AutoTokenizer,
            "cardiffnlp/twitter-xlm-roberta-base",
        ),
        "twitterxlmrobertabasesentiment": (
            AutoModel,
            AutoTokenizer,
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        ),
    }[args.btype]

    tokenizer = bert_type[1].from_pretrained(bert_type[2])
    model = bert_type[0].from_pretrained(bert_type[2], output_hidden_states=True)
    model.to(device).eval()

    dloc = "data/db_hbm/"

    with open(f"{dloc}prepared_data_w_dum_data_de_topics.p", "rb") as f:
        loaded_obj = pickle.load(f)

    # text = None
    # txt_processor = get_text_processor(htag=args.ht)
    # text = process_tweet(text, txt_processor)

    text_embed = []
    df = pd.DataFrame(loaded_obj)

    def get_text_embed(sample):

        sample_embed = {"tweet": None, "replies": [], "quotes": []}

        tweet_embed = get_text_embed_dict(
            sample["tweet"]["text"], model, tokenizer, device
        )
        sample_embed["tweet"] = tweet_embed

        for reply in sample["replies"]:
            sample_embed["replies"].append(
                get_text_embed_dict(reply["text"], model, tokenizer, device)
            )

        for quote in sample["quotes"]:
            sample_embed["quotes"].append(
                get_text_embed_dict(quote["text"], model, tokenizer, device)
            )

        return sample_embed

    tqdm.pandas()
    df["text_embed"] = df.progress_apply(get_text_embed, axis=1)

    text_embed = df["text_embed"].tolist()
    with open(f"features/{args.btype}_text_embed_w_de_topics.p", "wb") as f:
        pickle.dump(text_embed, f)


if __name__ == "__main__":

    args = parse_args()
    extract_txt_feats(args)
