import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd


df = pd.read_pickle("features/twitterxlmrobertabase_text_embed_w_dum_data.p")

df_topic = pd.read_pickle("data/db_hbm/prepared_data_w_dum_data.p")

sample = []
label_keys = {
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
i = 0
while True:
    data = {"embedding": [], "topic": []}
    data["embedding"] = df[i]["tweet"]["last"]
    data["topic"] = label_keys[df_topic[i]["topic"][0]]
    i += 1
    sample.append(data)
    if i == 10:
        break
df_emb = pd.DataFrame(sample)

tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)
x_embedded = tsne.fit_transform(np.array(df_emb["embedding"].tolist()))


labels = df_emb["topic"].values

plt.figure(figsize=(6, 5))
plt.title("title")
# sns.scatterplot(
#     x=df_emb["embedding"].values,
#     y=labels,
#     data=x_embedded,
# )
sns.scatterplot(
    x=x_embedded[:, 0],
    y=x_embedded[:, 1],
    hue=labels,
)


# plt.legend()
plt.show()
