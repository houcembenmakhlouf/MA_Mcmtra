import sqlite3
import numpy as np
import pandas as pd
from transformers import (
    BertTokenizer,
    BertModel,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
)
import argparse
import json
import torch
from helpers import get_bert_embeddings
from tqdm import tqdm
from sklearn.cluster import KMeans
import umap
import hdbscan
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


parser = argparse.ArgumentParser(description="Extract BERT Features")
parser.add_argument(
    "--btype",
    type=str,
    default="xlmrobertabase",
    help="bertbase | robertabase | xlmrobertabase",
)
args = parser.parse_args()


def get_hashtags_embedding():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_type = {
        "bertbase": (BertModel, BertTokenizer, "bert-base-uncased"),
        "robertabase": (RobertaModel, RobertaTokenizer, "roberta-base"),
        "xlmrobertabase": (XLMRobertaModel, XLMRobertaTokenizer, "xlm-roberta-base"),
    }[args.btype]

    tokenizer = bert_type[1].from_pretrained(bert_type[2])
    model = bert_type[0].from_pretrained(bert_type[2], output_hidden_states=True)
    model.to(device).eval()

    embed_dict = {"hashtag": [], "catavg": [], "sumavg": [], "2last": [], "last": []}

    # preprocess hashtags

    hashtags = prepare_dataset()
    for hashtag in tqdm(hashtags):
        # print(hashtag)

        (
            sent_word_catavg,
            sent_word_sumavg,
            sent_emb_2_last,
            sent_emb_last,
        ) = get_bert_embeddings(hashtag, model, tokenizer, device)

        embed_dict["hashtag"].append(hashtag)
        embed_dict["catavg"].append(sent_word_catavg.tolist())
        embed_dict["sumavg"].append(sent_word_sumavg.tolist())
        embed_dict["2last"].append(sent_emb_2_last.tolist())
        embed_dict["last"].append(sent_emb_last.tolist())

    json.dump(embed_dict, open("../features/%s_hashtags.json" % (args.btype), "w"))


def see_tweet_hastag():
    # db = dataset.connect(f"sqlite:///{dloc}mydatabase.db")
    dloc = "../data/db_hbm/"
    cnx = sqlite3.connect(f"{dloc}mydatabase.db")

    # consider only tweets with hashtags
    tweets_hashtags_df = pd.read_sql_query(
        "SELECT content FROM searchTweets WHERE tweetHashtags=='Wasserstoff'",
        cnx,
    )
    results_tweet = tweets_hashtags_df["content"].apply(lambda x: x.split(",")).tolist()
    for result in tqdm(results_tweet):
        print(result)
        print("\n")


def prepare_dataset():
    # db = dataset.connect(f"sqlite:///{dloc}mydatabase.db")
    dloc = "../data/db_hbm/"
    cnx = sqlite3.connect(f"{dloc}mydatabase.db")

    # consider only tweets with hashtags
    tweets_hashtags_df = pd.read_sql_query(
        "SELECT tweetHashtags FROM searchTweets WHERE tweetHashtags!=''", cnx
    )
    replies_hashtags_df = pd.read_sql_query(
        "SELECT tweetHashtags FROM repliesToTweet WHERE tweetHashtags!=''", cnx
    )
    quotes_hashtags_df = pd.read_sql_query(
        "SELECT tweetHashtags FROM quotesToTweet WHERE tweetHashtags!=''", cnx
    )
    results_tweet = (
        tweets_hashtags_df["tweetHashtags"].apply(lambda x: x.split(",")).tolist()
    )
    results_replies = (
        replies_hashtags_df["tweetHashtags"].apply(lambda x: x.split(",")).tolist()
    )
    results_quotes = (
        quotes_hashtags_df["tweetHashtags"].apply(lambda x: x.split(",")).tolist()
    )
    hashtags = []
    for result in tqdm(results_tweet):
        hashtags += result

    from collections import Counter

    words_to_count = (word for word in hashtags if word[:1].isupper())
    c = Counter(words_to_count)
    hashtags = [hashtag for hashtag, _ in c.most_common(1000)]
    print(hashtags)
    open("clusters.json", "w", encoding="utf8").write(
        json.dumps(hashtags, indent=4, ensure_ascii=False)
    )
    # with open("clusters.json", "w") as f:
    #     json.dumps(hashtags, f, ensure_ascii=False)
    # hashtags = set(hashtags)
    # hashtags = list(hashtags)
    print(len(hashtags))
    # hashtags = hashtags[0:1000]

    return hashtags


def generate_clusters():
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """
    with open("../features/xlmrobertabase_hashtags.json") as f:
        embed_dict = json.load(f)
    corpus_embeddings = np.array(embed_dict["last"])

    # corpus_embeddings = normalize(corpus_embeddings, norm="l2")
    hashtags = embed_dict["hashtag"]

    lowd_embeddings = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        metric="cosine",
        random_state=42,
    ).fit_transform(corpus_embeddings)
    # lowd_embeddings = PCA(n_components=10).fit_transform(corpus_embeddings)

    clusters = hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        min_samples=10,
        cluster_selection_method="eom",
    ).fit_predict(lowd_embeddings)

    cluster_hashtags = []
    for cluster in set(clusters):
        tmp_hashtags = [hashtags[i] for i, x in enumerate(clusters) if x == cluster]
        cluster_hashtags.append(tmp_hashtags)
    print(cluster_hashtags)

    clustered = clusters >= 0
    plt.scatter(
        lowd_embeddings[~clustered, 0],
        lowd_embeddings[~clustered, 1],
        color=(0.5, 0.5, 0.5),
        s=10,
        alpha=0.5,
        marker="x",
    )
    plt.scatter(
        lowd_embeddings[clustered, 0],
        lowd_embeddings[clustered, 1],
        c=clusters[clustered],
        s=4,
        cmap="Spectral",
    )
    plt.show()

    return clusters


def get_hashtags_clusters():
    with open("../features/xlmrobertabase_hashtags.json") as f:
        embed_dict = json.load(f)
    corpus_embeddings = embed_dict["last"]

    umap_embeddings = umap.UMAP(
        random_state=42,
    ).fit_transform(corpus_embeddings)
    kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(corpus_embeddings)
    plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=kmeans_labels,
        s=0.1,
        cmap="Spectral",
    )
    plt.show()
    # cluster_assignment = clustering_model.labels_
    # cluster_df = pd.DataFrame(hashtags, columns=["corpus"])
    # cluster_df["cluster"] = cluster_assignment
    # print(cluster_df)

    # clustered_words = [[] for i in range(num_clusters)]
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     clustered_words[cluster_id].append(hashtags[sentence_id])

    # for i, cluster in enumerate(clustered_words):
    #     print("Cluster ", i + 1)
    #     print(cluster)
    #     print("")


if __name__ == "__main__":
    see_tweet_hastag()
    # hashtags = prepare_dataset()
    # generate_clusters()
    # get_hashtags_embedding()
    # get_hashtags_clusters(hashtags)
