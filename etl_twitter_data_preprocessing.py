import pandas as pd
from dateutil import tz
from dateutil import parser
from datetime import timedelta
from transformers import pipeline
import os
from helpers import (
    clean_text
)

pd.options.mode.chained_assignment = None

# dataframe tweets
dataframe = pd.read_parquet("/home/daholive/Documents/twitter_ellection_brazil/datasource/raw/tweets.parquet")

###############################################################################
### pre processing
###############################################################################

# delete obsolete columns
dataframe.drop(inplace=True,columns=[
    "possibly_sensitive",
    "source",
    "annotations_normalized_text",
    "annotations_probability",
    "mentions_id",
    "mentions_username",
    "mentions_id",
    "urls_display_url",
    "urls_expanded",
    "urls_url",
    "referenced_tweets_type",
    "referenced_tweets_id",
    "cashtags_tag",
    "request_count",
    "annotations_type",
    "quote_count",
    "reply_count"
])

# filter portugueses tweets
dataframe = dataframe[dataframe["lang"]=="pt"]

# set sao paulo time zone
to_zone = tz.gettz('America/Sao_Paulo')

# adjust created_at to datetime
dataframe["created_at_tz"] = dataframe["created_at"].apply(lambda x: parser.isoparse(x).replace(tzinfo=to_zone).replace(tzinfo=None)) - timedelta(hours=3)

# create date from created_at
dataframe["dated_at_tz"] = dataframe["created_at_tz"].apply(lambda x: x.strftime("%Y-%m-%d"))

# cleaning text
dataframe["text_clean"] = dataframe["text"].map(clean_text)

# remove null values
dataframe = dataframe[dataframe["text_clean"]!='']

# remove new duplicated
dataframe.drop_duplicates(inplace=True,subset=["text"], keep="last")

# reindex dataframe 
dataframe = dataframe.reset_index().rename(columns={"index":"remove"})

# drop old index
dataframe.drop(["remove"],axis=1, inplace=True)

# classificador transformers / BERT using CUDA graphics for processing
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0)

# data chunks by text_clean
chunks = [dataframe["text_clean"][x:x+500] for x in range(0, dataframe["text_clean"].count(), 500)]

# processing sentiment list
classe_list = []
count=len(chunks)
for chunk in chunks:
    print(count)
    classe = classifier(list(chunk.values))
    classe_list.extend(classe)
    count -=1

# define bert level for each sentiment
dataframe["bert_sentiment_level"]  = [int(item["label"].split()[0]) for item in classe_list]

# sefine bert score for each sentiment
dataframe["bert_sentiment_score"]  = [round(item["score"],4) for item in classe_list]

# have retweets
dataframe["have_retweet"] = dataframe["retweet_count"].apply(lambda x: 1 if x > 0 else 0)

# have likes
dataframe["have_like"] = dataframe["like_count"].apply(lambda x: 1 if x > 0 else 0)

# save data preprocessing
dataframe.to_parquet("/home/daholive/Documents/twitter_ellection_brazil/datasource/trs/tweets_preprocessing.parquet")
