import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LinearDiscriminantAnalysis


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pycaret.classification import *
from helpers import (
    clean_text
)

dataframe = pd.read_csv("/home/daholive/Documents/twitter_ellection_brazil/datasource/TweetsWithTheme.csv", sep=",")

dataframe["sentiment"] = dataframe["sentiment"].map({
  "Negativo": 0,
  "Positivo": 1    
})

features = dataframe["tweet_text"].map(clean_text).values
labels = dataframe["sentiment"].values

vectorizer = TfidfVectorizer(
    min_df=0.004,
    max_df=0.7
)
processed_features = vectorizer.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=42)

text_classifier = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001)
text_classifier.fit(X_train, y_train)

# text_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
# text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))



# PYCARET
df_tfidf = pd.DataFrame(processed_features.toarray(), columns=vectorizer.get_feature_names(), index=dataframe.index)

df_setup = pd.concat([dataframe["sentiment"], df_tfidf], axis = 1)

%time pce_1 = setup(data = df_setup, target = 'sentiment', session_id = 5, train_size = 0.85, use_gpu = True)

models(internal=True)[['Name', 'GPU Enabled']]

best = compare_models()



# Please recompile with CMake option -DUSE_GPU=1










