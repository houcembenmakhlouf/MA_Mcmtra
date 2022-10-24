import json
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from collections import OrderedDict
import csv


def get_most_commun_hashtags(dloc):

    cnx = sqlite3.connect(f"{dloc}mydatabase.db")
    # consider only tweets with hashtags
    tweets_hashtags_df = pd.read_sql_query(
        "SELECT tweetHashtags FROM searchTweets WHERE tweetHashtags!=''", cnx
    )

    results_tweet = (
        tweets_hashtags_df["tweetHashtags"].apply(lambda x: x.split(",")).tolist()
    )

    hashtags = []
    for result in tqdm(results_tweet):
        hashtags += result
    print(len(hashtags))
    exit()
    words_to_count = (word for word in hashtags if word[:1].isupper())
    c = Counter(words_to_count)
    hashtags = [hashtag for hashtag, _ in c.most_common(1000)]

    with open(f"{cloc}clusters.json", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    all_hashtags = []
    for topic in json_data.values():
        all_hashtags = all_hashtags + topic

    main_list = list(set(hashtags) - set(all_hashtags))
    print(len(main_list))
    return main_list


def get_coocured_hashtag(dloc, cloc):

    all_hashtags = []

    with open(f"{cloc}clusters.json", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    for topic in json_data.values():
        all_hashtags = all_hashtags + topic

    cnx = sqlite3.connect(f"{dloc}mydatabase.db")

    # consider only tweets with hashtags
    tweets_hashtags_df = pd.read_sql_query(
        "SELECT tweetHashtags FROM searchTweets WHERE tweetHashtags!=''", cnx
    )

    results_tweet = (
        tweets_hashtags_df["tweetHashtags"].apply(lambda x: x.split(",")).tolist()
    )

    itemCount = 0

    with open(f"{cloc}records.csv", "w", encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",")
        sorted_dict_all = []
        for item in all_hashtags:
            coocurred = []
            for k in results_tweet:
                if item in k:
                    coocurred = coocurred + k
            # print(coocurred)

            c = Counter(coocurred)

            itemCount += 1
            print(itemCount)

            sorted_dict = OrderedDict(
                sorted(c.items(), key=lambda kv: kv[1], reverse=True)
            )

            sorted_dict_all.append(sorted_dict)
            fields = []
            rowData = []

            for x, v in sorted_dict.items():
                # if v >= five_prct_logic:
                fields.append(x)
                rowData.append(v)
            spamwriter.writerow(fields)
            spamwriter.writerow(rowData)

    return sorted_dict_all


def add_coocured_to_clusters(sorted_dict_all):

    dict = {
        "war": [],
        "automotiveIndustry": [],
        "corona": [],
        "sport": [],
        "politics": [],
        "fun": [],
        "history": [],
        "economics": [],
        "promotion": [],
        "social": [],
        "weather": [],
        "technology": [],
    }
    for hashtag_list in sorted_dict_all:

        list_of_hashtags = list(hashtag_list.items())
        origi_value_count = list(hashtag_list.items())[0][1]
        origi_item_value = list(hashtag_list.items())[0][0]
        ten_prct_logic = (origi_value_count * 40) / 100

        with open(f"{cloc}clusters_empowered.json", "r", encoding="utf-8") as jf:

            hashtags_to_add = []
            json_data = json.load(jf)
            for element in list_of_hashtags[1:]:
                if element[1] >= ten_prct_logic:
                    hashtags_to_add.append(element[0])

            new_list_to_replace = []
            for listTopic in json_data.values():
                if origi_item_value in listTopic:

                    new_list_to_replace += listTopic
                    new_list_to_replace += hashtags_to_add

                    # key where to put new list("war"...)
                    key_to_topic = list(json_data.keys())[
                        list(json_data.values()).index(listTopic)
                    ]

                    dict[key_to_topic] = new_list_to_replace

    with open(f"{cloc}clusters_empowered_cooccured.json", "w") as final:
        json.dump(dict, final, indent=4, ensure_ascii=False)

    # print(key_to_topic)
    # json.dump(new_list_to_replace, json_data[key_to_topic])


if __name__ == "__main__":
    dloc = "data/db_hbm/"
    cloc = "data/clusters_data/"
    # visualize most 1000 hashtags
    get_most_commun_hashtags(dloc)
    # hashtags = get_most_commun_hashtags(dloc)
    # print(hashtags)
    # get all hashtags with coocurrance
    # sorted_dict_all = get_coocured_hashtag(dloc, cloc)
    # update clusters based on new data
    # add_coocured_to_clusters(sorted_dict_all)
