import json
import pandas as pd
import sqlite3
import pickle
import re
from tqdm import tqdm
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


def get_clusters():
    with open("data/clusters_data/clusters_empowered_cooccured.json", "r") as j:
        data_dict = json.load(j)
    data_items = data_dict.items()
    data_list = list(data_items)
    clusters_df = pd.DataFrame(data_list)
    clusters_df.columns = ["topic", "data"]

    return clusters_df


# paper technique to preprocess
def get_text_processor(word_stats="twitter"):

    return TextPreProcessor(
        # terms that will be normalized , 'number','money', 'time','date', 'percent' removed from below list
        normalize=["url", "email", "phone", "user"],
        # terms that will be annotated
        annotate={
            "hashtag",
            "allcaps",
            "elongated",
            "repeated",
            "emphasis",
            "censored",
        },
        fix_html=True,  # fix HTML tokens
        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter=word_stats,
        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector=word_stats,
        # unpack_hashtags=htag,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        # spell_correct_elong=True,  # spell correction for elongated words
        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons],
    )


def prepare_dataset(dloc):

    # connect to db
    cnx = sqlite3.connect(f"{dloc}mydatabase.db")

    # consider only tweets with hashtags(lastest number of tweet was 150thousand)

    tweets_df = pd.read_sql_query(
        "SELECT * FROM searchTweets WHERE tweetHashtags!=''", cnx
    )
    replies_df = pd.read_sql_query(f"SELECT * FROM repliesToTweet", cnx)
    quotes_df = pd.read_sql_query(f"SELECT * FROM quotesToTweet", cnx)

    clusters_df = get_clusters()
    clusters_df = clusters_df.explode("data", ignore_index=True)

    tqdm.pandas()

    # tweets_df = tweets_df.head(10)

    def preprocess_text(text):
        # paper technique for preprocess
        # text_processor = get_text_processor()
        # # deep preprocess
        # text = text_processor.pre_process_doc(text)

        # clean_tweet = [
        #     word.strip() for word in text if not re.search(r"[^a-z0-9.,\s]+", word)
        # ]

        # clean_tweet = [
        #     word for word in clean_tweet if word not in ["rt", "http", "https", "htt"]
        # ]
        # clean_text = " ".join(clean_tweet)

        # remove links
        text = re.sub(r"http\S+", "", text)
        # remove linebreaks
        text = text.replace("\n", "")
        # remove mentions
        text = re.sub(r"(?:@[\w_]+)", "", text)
        # remove hashtags
        clean_text = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", "", text)

        return clean_text

    def create_samples(tweet):

        tweet_id = tweet["tweetId"]
        sample = {}
        replies_df_filtered = replies_df[replies_df["referenceTweetId"] == tweet_id]
        quotes_df_filtered = quotes_df[quotes_df["referenceTweetId"] == tweet_id]

        # if len(replies_df_filtered) >= 1 and tweet["tweetImage"] != None:
        if tweet["tweetImage"] != None:

            tweet_text = preprocess_text(tweet["content"])
            sample["tweet"] = {
                "text": tweet_text,
                "image": tweet["tweetImage"],
                "likesNbr": tweet["likeCount"],
            }

            replies = []
            for _, row in replies_df_filtered.iterrows():
                reply_text = preprocess_text(row["content"])
                replies.append(
                    {
                        "text": reply_text,
                        "image": row["tweetImage"],
                        "likesNbr": row["likeCount"],
                    }
                )

            sample["replies"] = replies

            quotes = []
            for _, row in quotes_df_filtered.iterrows():
                quote_text = preprocess_text(row["content"])
                quotes.append(
                    {
                        "text": quote_text,
                        "image": row["tweetImage"],
                        "likesNbr": row["likeCount"],
                    }
                )

            sample["quotes"] = quotes

            sample["tweetHashtags"] = tweet["tweetHashtags"].split(",")

            for hashtag in sample["tweetHashtags"]:
                # print("hashtag", hashtag)
                sample["topic"] = (
                    clusters_df[clusters_df["data"] == hashtag]["topic"]
                    .unique()
                    .tolist()
                )
                # print(sample["topic"])
            return sample

    tweets_df["samples"] = tweets_df.progress_apply(create_samples, axis=1)

    samples = tweets_df["samples"].tolist()
    # print(samples)
    # exit()
    # get only tweet with topic
    sample_with_topics = []
    for sample in tqdm(samples):
        try:
            if sample["topic"] != []:
                sample_with_topics.append(sample)
        except:
            pass

    with open(f"{dloc}prepared_data_w_omek.p", "wb") as f:
        pickle.dump(sample_with_topics, f)

    return sample_with_topics


if __name__ == "__main__":

    dloc = "data/db_hbm/"
    #  run preprocessing
    # samples = prepare_dataset(dloc)

    # for visualization of prepared data
    with open(f"{dloc}prepared_data_w_dum_data_fr_topics.p", "rb") as f:
        player = pickle.load(f)
        # print(player)
    x = 0
    rep_nbr = 0
    qte_nbr = 0
    img_nbr = 0
    for i in player:

        # print(i["tweet"]["text"])
        x += 1
        if i["replies"] != []:
            rep_nbr += len(i["replies"])
            # print(rep_nbr)
        if i["quotes"] != []:
            qte_nbr += len(i["quotes"])
    print("x", x)
    print("rep_nbr", rep_nbr)
    print("qte_nbr", qte_nbr)

    # get wanted classes
    # with open(f"{dloc}prepared_data_w_dum_data_fr_f.p", "rb") as f:
    #     player = pickle.load(f)
    # sample_with_topics_needed = []
    # for i in player:
    #     if "corona" in i["topic"] or "politics" in i["topic"] or "sport" in i["topic"]:
    #         sample_with_topics_needed.append(i)
    # with open(f"{dloc}prepared_data_w_dum_data_fr_topics.p", "wb") as f:
    #     pickle.dump(sample_with_topics_needed, f)
    # read
    # with open(f"{dloc}prepared_data_w_dum_data_fr_topics.p", "rb") as f:
    #     player = pickle.load(f)
    # print(player)
