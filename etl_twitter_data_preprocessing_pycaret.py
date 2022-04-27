import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pycaret.classification import *


dataframe = pd.read_csv("/home/daholive/Documents/twitter_ellection_brazil/datasource/TweetsWithTheme.csv", sep=",")

dataframe["sentiment"] = dataframe["sentiment"].map({
    "Negativo": 0,
    "Positivo": 1    
})

features = dataframe["tweet_text"].values
labels = dataframe["sentiment"].values

processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)
    
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('portuguese'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


df = pd.DataFrame(processed_features)
df_labels = pd.DataFrame(labels, columns=["target"])

df = pd.concat([df,df_labels], axis=1)
    
exp_clf101 = setup(
    data = df, 
    target = 'target',
    session_id = 5, 
    train_size = 0.80,
    fold=10,
    transformation = True,
    use_gpu = True,
    verbose=True) 












