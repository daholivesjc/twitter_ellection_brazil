import os, sys
import glob
import pandas as pd
from datetime import datetime, timedelta

files = glob.glob("/home/daholive/Documents/twitter_data/*.csv", recursive = True)

chunks = [files[x:x+100] for x in range(0, len(files), 100)]

def read_csv(file):
    
    return pd.read_csv(file, low_memory=False, sep=";")

datafrane_list = []
dataframe = pd.DataFrame()
count=len(chunks)
for lista in chunks:

    print(count)
    df = pd.concat(list(map(read_csv, lista)))
    del df["Unnamed: 0"]
    dataframe = pd.concat([dataframe, df])
    dataframe.drop_duplicates(inplace=True,subset=["twitter_id","text"])
    del df

    datafrane_list.append(dataframe)

    count -= 1

dataframe = pd.concat(datafrane_list)
dataframe.drop_duplicates(inplace=True,subset=["twitter_id","text"])

dataframe.to_parquet("/home/daholive/Documents/twitter_ellection_brazil/datasource/raw/tweets.parquet")

