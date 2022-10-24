import pickle
import pandas as pd


dloc = "data/db_hbm/"

with open(f"{dloc}prepared_data_w_dum_data_de_topics.p", "rb") as f:
    loaded_obj = pickle.load(f)

df = pd.DataFrame(loaded_obj)
df["topic"] = df["topic"].apply(", ".join)

print(df.groupby("topic").count())
