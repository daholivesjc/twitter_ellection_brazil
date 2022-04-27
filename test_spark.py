import numpy as np 
import pandas as pd
import unidecode
import pandas as pd 
import re
import nltk 
import string
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer
import pyspark 
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, IntegerType
import pyspark.sql.functions as F
from itertools import chain

import warnings
warnings.filterwarnings('ignore')

stemmer = RSLPStemmer()

def preprocessing(instancia):
    
    #reduz o termo ao seu radical, removendo afixos e vogais temáticas.
#---- stemmer = nltk.stem.RSLPStemmer()
    punct = string.punctuation
    trantab = str.maketrans(punct, len(punct)*' ')
    
    instancia = instancia.lower()
    instancia = re.sub('\d+', '', str(instancia)).replace("&gt;"," ").replace("&lt;"," ") 
    instancia = re.sub(r"https?:\/\/\S+","", instancia)
    instancia = re.sub(r"@[A-Za-z0-9\w]+","", instancia)
    instancia = re.sub(r"#[A-Za-z0-9\w]+","", instancia)
    instancia = re.sub('^RT ',' ',instancia)
    instancia = re.sub(r"http\S+", "", instancia) 
    
    instancia = instancia.translate(trantab).replace("\n"," ")
    
    instancia = unidecode.unidecode(instancia)

    # #Lista de  stopwords no idioma portugues
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    
    # #guarda no objeto palavras
    palavras = [stemmer.stem(i) for i in instancia.split() if not i in stopwords]
    
    palavras = [re.sub(r'(ha)\1+', r'\1',word) for word in palavras]
    palavras = [re.sub(r'(uha)\1+', r'\1',word) for word in palavras]
    palavras = [re.sub(r'(a)\1+', r'\1',word) for word in palavras]
    
    palavras = " ".join(palavras) \
        .strip() \
        .replace('"','') \
        .replace('.','') \
        .replace('-','') \
        .replace('_','') \
        .replace('*','') \
        .replace('>','') \
        .replace('<','') \
        .replace('!','') \
        .replace('?','') \
        .replace('[','') \
        .replace(']','') \
        .replace('\'','')

    return "-" if palavras.strip()=="" else palavras.strip()


# funcoes
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
    texto = ' '.join([stemmer.stem(word) for word in texto.split()])

    return "-" if texto.strip()=="" else texto.strip()


# SPARK INSTANCE
spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.executor.memory","4G") \
    .config("spark.driver.memory","4G") \
    .config("spark.executor.cores","12") \
    .getOrCreate()
    
# DADOS
dataframe = spark.read.options(
    delimiter=';',
    header='True').csv("/home/daholive/Documents/twitter_ellection_brazil/datasource/TweetsWithTheme_v2.csv")

###############################################################################
## PREPROCESSING WITH SPARK
###############################################################################
dataframe = dataframe.withColumn("sentiment_map", F.when(F.col("sentiment")=="Negativo", 0).otherwise(1))

rdd2 = dataframe.rdd.map(lambda x: (preprocessing(x.tweet_text),x.sentiment_map))

schema = StructType([       
    StructField('features', StringType(), True),
    StructField('label', StringType(), True),
])

# create metadata dataframe
df_features = spark.createDataFrame(rdd2, schema = schema)

df_features = df_features \
    .filter(F.col("features")!="-") \
    .dropDuplicates(subset = ['features'])

train = df_features.sampleBy("label", fractions={'0': 1, '1': 0.87}, seed=10)

# train.groupBy('label').count().show()

train.select(F.col('features')).show(2000,truncate=False)


###############################################################################
## FEATURE AND LABEL DEFINITION
###############################################################################
features = train.select('features').rdd.flatMap(lambda x: x).collect()

labels = train.select('label').rdd.flatMap(lambda x: x).collect()


###############################################################################
## TFIDF
###############################################################################
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Unigram Counts
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=0.00001, max_df=0.9, max_features=100000)
unigram_vectorizer.fit(features)
X_train_unigram = unigram_vectorizer.transform(features)

# Unigram Tf-Idf
unigram_tf_idf_transformer = TfidfTransformer()
unigram_tf_idf_transformer.fit(X_train_unigram)
X_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_train_unigram)

# Bigram Counts
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),  min_df=0.0001, max_df=0.9, max_features=100000)
bigram_vectorizer.fit(features)
X_train_bigram = bigram_vectorizer.transform(features)

# Bigram Tf-Idf
bigram_tf_idf_transformer = TfidfTransformer()
bigram_tf_idf_transformer.fit(X_train_bigram)
X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)

# Trigram Counts
trigram_vectorizer = CountVectorizer(ngram_range=(1, 3),  min_df=0.0001, max_df=0.9, max_features=100000)
trigram_vectorizer.fit(features)
X_train_trigram = trigram_vectorizer.transform(features)

# Bigram Tf-Idf
trigram_tf_idf_transformer = TfidfTransformer()
trigram_tf_idf_transformer.fit(X_train_trigram)
X_train_trigram_tf_idf = trigram_tf_idf_transformer.transform(X_train_trigram)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

scaler = MaxAbsScaler()
rescaledX = scaler.fit_transform(X_train_unigram_tf_idf)    

scaler = Normalizer().fit(X_train_unigram_tf_idf)
normalizedX = scaler.transform(X_train_unigram_tf_idf)

scaler = StandardScaler(with_mean=False).fit(X_train_unigram_tf_idf)
standardX = scaler.transform(X_train_unigram_tf_idf)

binarizer = Binarizer(threshold = 0.2).fit(X_train_unigram_tf_idf)
binaryX = binarizer.transform(X_train_unigram_tf_idf)

###############################################################################
## MODEL TESTS  LogisticRegression
###############################################################################
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.7, random_state=42
    )

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')
train_and_show_scores(X_train_trigram, y_train, 'Trigram Counts')
train_and_show_scores(X_train_trigram_tf_idf, y_train, 'Trigram Tf-Idf')

train_and_show_scores(rescaledX, y_train, 'Trigram Tf-Idf')
train_and_show_scores(normalizedX, y_train, 'Trigram Tf-Idf')
train_and_show_scores(standardX, y_train, 'Trigram Tf-Idf')
train_and_show_scores(binaryX, y_train, 'Trigram Tf-Idf')

"""
Unigram Counts
Train score: 0.84 ; Validation score: 0.75

Unigram Tf-Idf
Train score: 0.81 ; Validation score: 0.76

Bigram Counts
Train score: 0.99 ; Validation score: 0.77

Bigram Tf-Idf
Train score: 0.9 ; Validation score: 0.76
"""


###############################################################################
## MODEL TESTS  MultinomialNB
###############################################################################
from sklearn.naive_bayes import MultinomialNB
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.70, random_state=42
    )

    clf = MultinomialNB()
    
    param_grid = {'alpha': [0.001,0.1,0.5,0.9,1,10] }

    kfold = StratifiedKFold(n_splits = 5, shuffle = True)
    
    grid_search = GridSearchCV(
        MultinomialNB(), 
        param_grid=param_grid, 
        cv = kfold,
        n_jobs=-1,
        scoring='accuracy')
    grid_result = grid_search.fit(X_train, y_train)
    
    # clf.fit(X_train, y_train)
    train_score = grid_result.score(X_train, y_train)
    valid_score = grid_result.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')
train_and_show_scores(X_train_trigram, y_train, 'Trigram Counts')
train_and_show_scores(X_train_trigram_tf_idf, y_train, 'Trigram Tf-Idf')






###############################################################################
## MODEL TESTS  GradientBoostingClassifier
###############################################################################
from sklearn.ensemble import GradientBoostingClassifier
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.70, random_state=42
    )

    model = GradientBoostingClassifier()
    
    n_estimators = [100]
    learning_rate = [1.0]
    max_depth = [3, 7, 9]
    
    grid = dict(
        learning_rate=learning_rate, 
        n_estimators=n_estimators, 
        max_depth=max_depth)

    kfold = StratifiedKFold(n_splits = 5, shuffle = True)
    
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=grid, 
        n_jobs=-1, 
        cv=kfold, 
        scoring='accuracy',
        error_score=0)
    grid_result = grid_search.fit(X, y)
    
    # clf.fit(X_train, y_train)
    train_score = grid_result.score(X_train, y_train)
    valid_score = grid_result.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')










###############################################################################
## MODEL TESTS  LogisticRegression TEST
###############################################################################
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.70
    )

    clf = LogisticRegression()
    
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    
    kfold = StratifiedKFold(n_splits = 5, shuffle = True)
    
    grid_search = GridSearchCV(
        estimator=clf, 
        param_grid=grid, 
        n_jobs=-1, 
        cv=kfold, 
        scoring='f1',
        error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    
    # clf.fit(X_train, y_train)
    train_score = grid_result.score(X_train, y_train)
    valid_score = grid_result.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')

y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')







###############################################################################
## MODEL TESTS  LogisticRegression - HYPER PARAMETERS
###############################################################################
from sklearn.model_selection import RandomizedSearchCV
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y, random_state=42
    )
    
    clf = LogisticRegression()
    
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    
    kfold = StratifiedKFold(n_splits = 10, shuffle = True)
    
    random_search_cv = RandomizedSearchCV(
        estimator=clf,
        param_distributions=grid,
        cv=kfold,
        n_iter=50
    )
    random_search_cv_result = random_search_cv.fit(X_train, y_train)
    
    # clf.fit(X_train, y_train)
    train_score = random_search_cv_result.score(X_train, y_train)
    valid_score = random_search_cv_result.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')
""" 
Unigram Counts
Train score: 0.79 ; Validation score: 0.76

Unigram Tf-Idf
Train score: 0.81 ; Validation score: 0.76

Bigram Counts
Train score: 0.85 ; Validation score: 0.77

Bigram Tf-Idf
Train score: 0.86 ; Validation score: 0.77

StratifiedKFold
Unigram Counts
Train score: 0.79 ; Validation score: 0.76

Unigram Tf-Idf
Train score: 0.81 ; Validation score: 0.76

Bigram Counts
Train score: 0.99 ; Validation score: 0.77

Bigram Tf-Idf
Train score: 1.0 ; Validation score: 0.77
"""



###############################################################################
## MODEL TESTS  RidgeClassifier - HYPER PARAMETERS
###############################################################################
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = RidgeClassifier()
    
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    grid = dict(alpha=alpha)
    
    random_search_cv = RandomizedSearchCV(
        estimator=clf,
        param_distributions=grid,
        cv=5,
        n_iter=50
    )
    random_search_cv_result = random_search_cv.fit(X_train, y_train)
    
    # clf.fit(X_train, y_train)
    train_score = random_search_cv_result.score(X_train, y_train)
    valid_score = random_search_cv_result.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')



















###############################################################################
## MODEL TESTS  SGDClassifier
###############################################################################
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')
"""
Unigram Counts
Train score: 0.82 ; Validation score: 0.75

Unigram Tf-Idf
Train score: 0.79 ; Validation score: 0.76

Bigram Counts
Train score: 0.95 ; Validation score: 0.75

Bigram Tf-Idf
Train score: 0.82 ; Validation score: 0.76
"""


###############################################################################
## MODEL TESTS  RidgeClassifier
###############################################################################
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')
"""
Unigram Counts
Train score: 0.86 ; Validation score: 0.74

Unigram Tf-Idf
Train score: 0.84 ; Validation score: 0.75

Bigram Counts
Train score: 0.99 ; Validation score: 0.72

Bigram Tf-Idf
Train score: 0.96 ; Validation score: 0.76
"""




###############################################################################
## MODEL TESTS RandomForestClassifier
###############################################################################
from sklearn.ensemble import RandomForestClassifier
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.70, stratify=y
    )

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')



###############################################################################
## MODEL TESTS RandomForestClassifier - HYPER PARAMETERS
###############################################################################
from sklearn.ensemble import RandomForestClassifier
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.70, stratify=y
    )

    clf = RandomForestClassifier()
    # clf.fit(X_train, y_train)
    
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']
    
    grid = dict(n_estimators=n_estimators,max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    
    # score
    train_score = grid_result.score(X_train, y_train)
    valid_score = grid_result.score(X_valid, y_valid)
    
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')





###############################################################################
## MODEL TESTS BaggingClassifier - HYPER PARAMETERS
###############################################################################
from sklearn.ensemble import BaggingClassifier
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = BaggingClassifier()
    # clf.fit(X_train, y_train)
    # train_score = clf.score(X_train, y_train)
    # valid_score = clf.score(X_valid, y_valid)
    n_estimators = [10, 100, 1000]
    
    grid = dict(n_estimators=n_estimators)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    
    # score
    train_score = grid_result.score(X_train, y_train)
    valid_score = grid_result.score(X_valid, y_valid)
    
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')












###############################################################################
## MODEL TESTS - RidgeClassifier - HYPER PARAMETERS
###############################################################################
def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = RidgeClassifier()
    # clf.fit(X_train, y_train)
    # train_score = clf.score(X_train, y_train)
    # valid_score = clf.score(X_valid, y_valid)
    
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # define grid search
    grid = dict(alpha=alpha)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    
    # score
    train_score = grid_result.score(X_train, y_train)
    valid_score = grid_result.score(X_valid, y_valid)
    
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')

y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')

"""
Unigram Counts
Train score: 0.86 ; Validation score: 0.74

Unigram Tf-Idf
Train score: 0.84 ; Validation score: 0.76

Bigram Counts
Train score: 1.0 ; Validation score: 0.75

Bigram Tf-Idf
Train score: 0.98 ; Validation score: 0.77
"""




###############################################################################
## MODEL TESTS - KNeighborsClassifier - HYPER PARAMETERS
###############################################################################
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = KNeighborsClassifier()
    # clf.fit(X_train, y_train)
    # train_score = clf.score(X_train, y_train)
    # valid_score = clf.score(X_valid, y_valid)

    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']

    # define grid search
    grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X_train, y_train)
    
    # score
    train_score = grid_result.score(X_train, y_train)
    valid_score = grid_result.score(X_valid, y_valid)
    
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')

y_train = labels

train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')














###############################################################################
## HYPER PARAMETERS TUNNING
###############################################################################
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

X_train = X_train_bigram_tf_idf


# Phase 1: loss, learning rate and initial learning rate

clf = SGDClassifier()

distributions = dict(
    loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    learning_rate=['optimal', 'invscaling', 'adaptive'],
    eta0=uniform(loc=1e-7, scale=1e-2)
)

random_search_cv = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distributions,
    cv=5,
    n_iter=50
)
random_search_cv.fit(X_train, y_train)
print(f'Best params: {random_search_cv.best_params_}')
print(f'Best score: {random_search_cv.best_score_}')


# Phase 2: penalty and alpha

clf = SGDClassifier()

distributions = dict(
    penalty=['l1', 'l2', 'elasticnet'],
    alpha=uniform(loc=1e-6, scale=1e-4)
)

random_search_cv = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distributions,
    cv=5,
    n_iter=50
)
random_search_cv.fit(X_train, y_train)
print(f'Best params: {random_search_cv.best_params_}')
print(f'Best score: {random_search_cv.best_score_}')


# Phase 3: penalty, alpha, loss, learning rate and initial learning rate

clf = SGDClassifier()

distributions = dict(
    loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    learning_rate=['optimal', 'invscaling', 'adaptive'],
    eta0=uniform(loc=1e-7, scale=1e-2),
    penalty=['l1', 'l2', 'elasticnet'],
    alpha=uniform(loc=1e-6, scale=1e-4)
)

random_search_cv = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distributions,
    cv=5,
    n_iter=50
)
random_search_cv.fit(X_train, y_train)
print(f'Best params: {random_search_cv.best_params_}')
print(f'Best score: {random_search_cv.best_score_}')












