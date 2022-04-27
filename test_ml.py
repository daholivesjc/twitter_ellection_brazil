import numpy as np 
import unidecode
import pandas as pd 
import re
import nltk 
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pycaret.classification import *
from nltk.stem.rslp import RSLPStemmer
from helpers import (
    clean_text
)
from time import time
import warnings
warnings.filterwarnings('ignore')

lemmatizer = RSLPStemmer()

dataframe = pd.read_csv("/home/daholive/Documents/twitter_ellection_brazil/datasource/TweetsWithTheme.csv", sep=",")

dataframe["sentiment"] = dataframe["sentiment"].map({
    "Negativo": 0,
    "Positivo": 1    
})

# stop_words_br = list(set(stopwords.words("portuguese")))
# stop_words_br_no_accent = [unidecode.unidecode(word) for word in list(set(stopwords.words("portuguese")))]

def clean_text(texto):
    
    punct = string.punctuation # Cria uma tabela de tradução
    trantab = str.maketrans(punct, len(punct)*' ') # Todo simbolo da pontuação e substituido por um espaço
    
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030""]+", re.UNICODE)
    
    texto = texto.lower()
    texto = re.sub('\d+', '', str(texto)).replace("&gt;"," ").replace("&lt;"," ") 
    texto = re.sub(r"https?:\/\/\S+","", texto)
    texto = re.sub(r"@[A-Za-z0-9\w]+","", texto)
    texto = re.sub(r"#[A-Za-z0-9\w]+","", texto)
    texto = re.sub('^RT ',' ',texto)
    texto = texto.translate(trantab).replace("\n"," ")
    # texto = re.sub(emoj, '', texto).replace("“"," ").replace("”"," ").strip().lower()
    texto = texto.replace("“"," ").replace("”"," ")
    texto = unidecode.unidecode(texto)
    texto = ' '.join([word for word in texto.split() if word not in list(set(stopwords.words("portuguese")))])
    texto = ' '.join([word for word in texto.split() if word.isalnum()])
    texto = ' '.join([re.sub(r'([a-z])\1+', r'\1',word) for word in texto.split()])
    texto = ' '.join([re.sub(r'(ha)\1+', r'\1',word) for word in texto.split()])
    texto = ' '.join([re.sub(r'(uha)\1+', r'\1',word) for word in texto.split()])
    texto = ' '.join([lemmatizer.stem(word) for word in texto.split()])

    return texto.strip()


features = dataframe["tweet_text"].map(clean_text).values
labels = dataframe["sentiment"].values

"""
max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:

max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
max_df = 25 means "ignore terms that appear in more than 25 documents".
The default max_df is 1.0, which means "ignore terms that appear in more than 100% of the documents". Thus, the default setting does not ignore any terms.

min_df is used for removing terms that appear too infrequently. For example:

min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
min_df = 5 means "ignore terms that appear in less than 5 documents".
The default min_df is 1, which means "ignore terms that appear in less than 1 document". Thus, the default setting does not ignore any terms.
"""

"""
vectorizer = TfidfVectorizer(
    min_df=0.004,
    max_df=0.7
)
"""
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


X_train, X_test, y_train, y_test = train_test_split(
    features, 
    labels, 
    test_size=0.33, 
    random_state=42)


# criando o modelo usando pipeline
model = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(
        ngram_range=(1,2), 
        analyzer= 'word',
        max_features=10000
    )),
    ('rd', RidgeClassifier(
        class_weight='balanced', 
        normalize=True, 
        solver='auto')
    ),
])


# Tunando hiperparâmetros com 5-fold cross-validation e pipelines
#  'tfidf__max_features': [250, 500, 1000, 10000],
parameters = {
    'tfidf__min_df': [0.01, 0.001, 0.0001],
    'tfidf__max_df': [0.25, 0.50, 0.75, 1.0],
    'tfidf__norm': ['l1', 'l2'],
    'rd__alpha': [0.0001, 0.001, 0.1, 0.5, 0.9, 1, 10]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(model, param_grid=parameters, cv=kfold, n_jobs=-1)

grid.fit(X_train,y_train)

train_score = grid.score(X_train, y_train)

test_score = grid.score(X_test, y_test)

print("Train score: {}".format(train_score))
print("Test score: {}".format(test_score))

grid.best_params_ 

print('Melhores parâmetros: ',grid.best_params_,'Melhor score: ', grid.best_score_,) 

# model.get_params()










X_train, X_test, y_train, y_test = train_test_split(
    features, 
    labels, 
    test_size=0.33, 
    random_state=42)
# treinando o modelo

model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)

# avaliando o modelo
test_score = model.score(X_test, y_test)

print("Train score: {}".format(train_score))
print("Test score: {}".format(test_score))

vectorizer  = TfidfVectorizer(
    min_df=10,
    max_df=0.8,
    ngram_range=(1,2),
    analyzer= 'word').fit(features)

#Aplicando nos dados de treinamento
vec_X_train = vectorizer.transform(X_train)

#Aplicando nos dados de teste
vec_X_test = vectorizer.transform(X_test)

param_grid = {
    'alpha': [0.001,0.1,0.5,0.9,1,10]
}

grid_search = GridSearchCV(MultinomialNB(), param_grid, cv = 15 )#cv = cross validation
    
grid_search.fit(vec_X_train,y_train)

grid_search.best_params_ 

print('Melhores parâmetros: ',grid_search.best_params_,'Melhor score: ', grid_search.best_score_,) 


   

    





modelo = MultinomialNB(alpha=1).fit(vec_X_train,y_train)

novos_tweets = [  'Fora Temer!!! Golpista!',
                  'Corrupção com Aécio Neves!',
                  'O governo está investindo em educação',
                  'Vamos todos andar de bibicleta hoje e sermos saudáveis!',
                  'A qualidade do ensino em Minas Gerais é excelente!',
                  'Fora Pimentel!! Golpista!',
                  'Muitas cidades estão decretando calamidade administrativa!',
                  'A febre amarela está matando muitas pessoas e o governo não faz nada para ajudar!',
                  'Minas Gerais tem  surto de febre amarela',
                  'Eu andei de helicoptero',
                  'A falta de recursos é consequência da corrupção'
                
                ]

novos_tweets = [clean_text(i) for i in novos_tweets]

def probabilidade_tweet(novos_tweets):
    vectorizer_novos_tweets = vectorizer.transform(novos_tweets)
    # Fazendo a classificação com o modelo treinado.
    for tweet, classificacao ,probabilidade in zip (novos_tweets,modelo.predict(vectorizer_novos_tweets),modelo.predict_proba(vectorizer_novos_tweets).round(2)):
        print('Sentença: ',tweet ,'\n',"Classificaçaõ: ", classificacao ,'\n', 'Probabilidades','\n','Neg', ' | ' ,'Nt', '|','Posi' ,'\n',probabilidade,'\n')
    
probabilidade_tweet(novos_tweets)
    
    
    
    
    
    





