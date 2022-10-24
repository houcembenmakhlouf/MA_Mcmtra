# Multilingual and Cross-Modal Embedded Representation for Tweets Aggregation
This repository provides code for this master thesis:
## Abstract
Twitter data has been useful in different tasks such as topic classification and sentiment analysis. The Twitter’s raw data, however, is multilingual and presents noise and is shorttext. In order to catch useful information from tweet, the high-dimensional raw data should be encoded in lower-dimensional representations. In recent years, different aspects were the center of research to enhance the meaningfulness of the tweet representations like including different modalities of data such as: text, audio, images, and videos. However, there are not many works about multilingual multimodal representations due to lack of large high-quality datasets. In this thesis, I propose a new model to learn multilingual and multimodal representations of individual tweets by leveraging a topic classification task. My model not only uses traditional modalities, like language and vision, but also profits from Twitter-specific modalities such as replies and quotes. My thesis first shows that adding replies and quotes texts and images features to the tweet features improves the model’s accuracy by at least 3%. My work also shows that combining multimodal and multilingual features by using attention mechanisms increases the meaningfulness of tweet’s related input (tweet, its replies and quotes) representation by preserving the semantic closeness of the input data and increasing the accuracy with additional 5%. The results demonstrate how different multimodal and multilingual embeddings within each document can be learned in an aligned manner in order to push similar classes close to each other. I anticipate my thesis to be another step for more works that want to combine cross modals for Twitter data in several languages in order to enhance a tweet representation.

## Data collection
This project resulted in "MCMTRA dataset". Refer to https://github.com/houcembenmakhlouf/MA_osint_gatherer repository which provides a platform for data collection process from Twitter API. 

## Embeddings extraction
the project uses pre-trained models ```XLM-RoBERTa``` (https://huggingface.co/docs/transformers/model_doc/xlm-roberta) and ```ResNet-50``` (https://arxiv.org/abs/1512.03385) for extracting textual and visual features respectively. The embbeding extraction process is insured by running the files ```extract_txt_feats.py``` and ```extract_img_feats``` under the folder ```feature_extraction```.


## Models
The table below show all models that were considered in this project. 
<center>
  
| Model | Description |
| ------------- | ------------- |
hbmnet_baseline_t | multilingual baseline model for text|
hbmnet_baseline_tv | multilingual baseline model for text and image|
hbmnet | multimodal multilingual model with replies and quotes|
hbmnet_w_likes | same as ”hbmnet” with number of likes feature|
hbmnet_w_self_attention | same as ”hbmnet” with self attention module|
hbmnet_w_attention | same as ”hbmnet” with attention between tweet and interactions|
hbmnet_attention_likes | same as ”hbmnet” with both attention mechanisms and number of likes feature|
hbmnet_w_double_attention |same as ”hbmnet” with both attention mechanisms|

</center>

The best model is shown in the figure below:

<center>
  
<p align="center">
  <img src="https://user-images.githubusercontent.com/45092804/197520335-5e307611-2ea4-4d70-bc9f-d8c4c7645b79.png" width="600" />
</p>

</center>

## Setting up an environment
This framework is built using Python 3.6 and relies on the PyTorch 1.4.0+. The following command installs all necessary packages:
```
pip3 install -r requirements.txt
```

##training 
We provided scripts for training the proposed model on "MCMTRA dataset" . You can start training by running the ```train_cv.py```(with cross-validation)
For each experiment, a separate folder is created in the ./exps with Tensorboard logs, text logs, visualization and model's checkpoints. You can specify another path in the [config.yml](https://github.com/houcembenmakhlouf/MA_MCMTRA/blob/master/config.yaml) (see EXPS_PATH variable).






This project uses the paper (Paper is available on arXiv: https://arxiv.org/abs/2106.08829) as reference.
# A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods

The repository for the splits and code used in the paper
> Gullal S. Cheema, Sherzod Hakimov, Eric Müller-Budack and Ralph Ewerth “A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment Analysis Methods“, 
*Proceedings of the 2021 Workshop on Multi-Modal Pre-Training for Multimedia Understanding (MMPT ’21), August 21, 2021,Taipei, Taiwan.

Paper is available on arXiv: https://arxiv.org/abs/2106.08829

## Data and Environment
- Available at https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
- Download and extract images and texts into separate folders in `data/mvsa_single/` and `data/mvsa_multiple/`
- Create an environment using environment.yml with conda
## Splits
- 10 Fold Train/Val/Test splits provided in data/ for MVSA-single and MVSA-multiple.
- valid_pairlist.txt format is `file_id (filename), multimodal label, text label, image label`
- 0 (Neutral), 1 (Positive), 2 (Negative)
- Split file rows point to the line number in valid_pairlist.txt (0-indexed)
- `multimodal label` is used for training and evaluating all the models.

## Extract Features
- Download pretrained models to `pre_trained` : [places](https://drive.google.com/file/d/1ARP8GS5LMGYc8T8lFTuYkBl9I9kJoIiL/view?usp=sharing), [emotion](https://drive.google.com/file/d/1sWx3ze8XfZEGf-kPcmiYpY9EOzugdzgu/view?usp=sharing).
- Download face expression features into `features` folder from [mvsa_single](https://drive.google.com/file/d/1akwt0-8fqea2WinTDWAk68qRDkYqEEiR/view?usp=sharing), [mvsa_multiple](https://drive.google.com/file/d/1h6A8KjhVEy2QSEBGQS8kH2HQxEa__a2S/view?usp=sharing) 
- Extract image features: `python feature_extraction/extract_img_feats.py --vtype imagenet --mvsa single --ht False`
- Extract text features: `python feature_extraction/extract_txt_feats.py --btype robertabase --mvsa single --ht True`

## Train and evaluate models
- With one type of visual feature and BERT feature: `python train/multi_mlp_2mod.py --vtype clip --ttype clip --mvsa single --ht True --bs 32 --split 1`


```
@inproceedings{DBLP:conf/mir/CheemaHME21,
  author    = {Gullal S. Cheema and
               Sherzod Hakimov and
               Eric M{\"{u}}ller{-}Budack and
               Ralph Ewerth},
  editor    = {Bei Liu and
               Jianlong Fu and
               Shizhe Chen and
               Qin Jin and
               Alexander G. Hauptmann and
               Yong Rui},
  title     = {A Fair and Comprehensive Comparison of Multimodal Tweet Sentiment
               Analysis Methods},
  booktitle = {MMPT@ICMR2021: Proceedings of the 2021 Workshop on Multi-Modal Pre-Training
               for Multimedia Understanding, Taipei, Taiwan, August 21, 2021},
  pages     = {37--45},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3463945.3469058},
  doi       = {10.1145/3463945.3469058},
  timestamp = {Wed, 06 Oct 2021 14:51:08 +0200},
  biburl    = {https://dblp.org/rec/conf/mir/CheemaHME21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
