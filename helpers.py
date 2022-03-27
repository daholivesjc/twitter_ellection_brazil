from settings import BEARER_TOKEN, TWITTER_API_RECENT_SEARCH
import requests
import json
import pickle
from datetime import datetime, timedelta
import sys, os
import glob
import pandas as pd
import string
import re
import torch
from transformers import ( 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from textblob import TextBlob
from flair.data import Sentence
import time
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem.rslp import RSLPStemmer
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation # NMF
from time import time
import unidecode
nlp = spacy.load("pt_core_news_lg")

stop_words_br = list(set(stopwords.words("portuguese")))
stop_words_br_no_accent = [unidecode.unidecode(word) for word in list(set(stopwords.words("portuguese")))]

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
# bearer_token = os.environ.get("BEARER_TOKEN")

def twitter_bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r


def get_twitter_params(query, end_time):
    
    tweet_fields = [
        "author_id",
        "text",
        "created_at",
        "entities",
        "geo",
        "in_reply_to_user_id",
        "lang,possibly_sensitive",
        "referenced_tweets",
        "source",
        "public_metrics"]
 
    query_params = {
        "query": query,
        "tweet.fields": ",".join(tweet_fields), 
        "end_time": end_time,
        "max_results": 100
    }
        
    return query_params


def get_recents_tweets(logger, end_time, limit=10, query=""):
    
    logger.info(query)
    
    params = get_twitter_params(query, end_time)
    bearer_oauth = twitter_bearer_oauth
    search_url = TWITTER_API_RECENT_SEARCH
    
    data_list = []
    data_dict = {}
    has_more = True
    count = 1
    while has_more:

        if count <= limit:
        
            r = requests.get(search_url, auth=bearer_oauth, params=params)
            response_dict = json.loads(r.text)
            
            logger.info(r.status_code)
            
            if response_dict["meta"]["result_count"] > 0:
                
                if response_dict["meta"].get('next_token',''):
                    
                    next_token = response_dict["meta"]["next_token"]
                    params["pagination_token"] = next_token
                
                    data_dict = {
                        "query": query,
                        "data": response_dict["data"],
                        "request_count": count
                    }
                
                    data_list.extend([data_dict])
                
                    count += 1
                
                else:
                    has_more = False
            
            else:
                has_more = False
        
        else:
            has_more = False
    
    return data_list


def save_tweets_file(tweet_list):

    path = os.path.abspath(os.path.join('..', '')) + "/kali/twitter/twitter_ellection_brazil"
    
    timestamp = int((datetime.utcnow() - timedelta(minutes = 4)).timestamp()*1e3)
    
    with open(f"{path}/tweets/twitter_{timestamp}.lst", "wb") as fp:
        pickle.dump(tweet_list, fp)
      

def save_tweets_dataframe(dataframe, path_dataframe):

    path = os.path.abspath(os.path.join('..', '')) + "/kali/twitter/twitter_ellection_brazil"
    
    timestamp = int((datetime.utcnow() - timedelta(minutes = 4)).timestamp()*1e3)
    
    # dataframe.to_pickle(f"{path}/dataframe/dataframe_{timestamp}.pkl")
    
    # dataframe.to_pickle(f"{path_dataframe}/dataframe_{timestamp}.pkl")
    
    dataframe.to_csv(f"{path_dataframe}/dataframe_{timestamp}.csv",encoding="utf-8", sep=";")
    
    print("Dataframe Tweets saved successfully!")
        
def delete_tweets_file(files):
    
    for file in files:
        
        try:
            os.remove(file)
        except:
            pass

        
def read_tweet_files():

    path = os.path.abspath(os.path.join('..', '')) + "/kali/twitter/twitter_ellection_brazil"
    
    files = glob.glob(f'{path}/tweets/*.lst', recursive = True)
    
    if not files:
        
        print("There are no files to be processed!")
        sys.exit()

    data_list = []
    data = {}
    for file in files:
        print(file)
        
        with open(file, "rb") as fp: 
            try:     
                file_data = pickle.load(fp)
                data_list.extend(file_data)
            except:
                pass
            
    twitter_list = []
    twitter_list_mentions = []
    twitter_list_annotations = []
    twitter_list_urls = []
    twitter_list_hashtags = []
    twitter_list_cashtags = []
    twitter_list_public_metrics = []
    twitter_list_referenced_tweets = []

    for twitter in data_list:
        for tweet in twitter["data"]:
        
            data = {
                "author_id": tweet["author_id"],
                "created_at": tweet["created_at"],
                "twitter_id": tweet["id"],
                "lang": tweet["lang"],
                "possibly_sensitive": tweet["possibly_sensitive"],
                "source": tweet["source"],
                "text": tweet["text"],
                "query": twitter["query"],
                "request_count": twitter["request_count"]
            }
               
            if tweet.get('referenced_tweets',''):
        
                for refer in tweet["referenced_tweets"]:
                    
                    referenced_tweets = {
                        "twitter_id": tweet["id"],
                        "referenced_tweets_type": refer["type"],
                        "referenced_tweets_id": refer["id"]
                    }
                    
                    twitter_list_referenced_tweets.append(referenced_tweets)
            
            if tweet.get('public_metrics',''):
                
                public_metrics = {
                    "twitter_id": tweet["id"],
                    "like_count": tweet["public_metrics"]["like_count"],
                    "quote_count": tweet["public_metrics"]["quote_count"],
                    "reply_count": tweet["public_metrics"]["reply_count"],
                    "retweet_count": tweet["public_metrics"]["retweet_count"]
                }
                
                twitter_list_public_metrics.append(public_metrics)
        
        
            if tweet.get('entities',''):
                
                if tweet["entities"].get('annotations',''):
                    
                    for annot in tweet["entities"].get('annotations',''):
                        
                        annotations = {
                            "twitter_id": tweet["id"],
                            "annotations_normalized_text": annot["normalized_text"],
                            "annotations_probability": annot["probability"],
                            "annotations_type": annot["type"]
                        }
                        
                        twitter_list_annotations.append(annotations)
                        
                if tweet["entities"].get('mentions',''):
                    
                    for ment in tweet["entities"].get('mentions',''):
                        
                        mentions = {
                            "twitter_id": tweet["id"],
                            "mentions_id": ment["id"],
                            "mentions_username": ment["username"]
                        }
                        
                        twitter_list_mentions.append(mentions)
        
                if tweet["entities"].get('urls',''):
            
                    for url in tweet["entities"].get('urls',''):
                        
                        urls = {
                            "twitter_id": tweet["id"],
                            "urls_display_url": url["display_url"],
                            "urls_expanded": url["expanded_url"],
                            "urls_url": url["url"]
                        }
            
                        twitter_list_urls.append(urls)
                        
                
                if tweet["entities"].get('hashtags',''):
            
                    for hashtag in tweet["entities"].get('hashtags',''):
                        
                        hashtags = {
                            "twitter_id": tweet["id"],
                            "hashtags_tag": hashtag["tag"]
                        }
            
                        twitter_list_hashtags.append(hashtags)
                
                if tweet["entities"].get('cashtags',''):
            
                    for cash in tweet["entities"].get('cashtags',''):
                        
                        cashtags = {
                            "twitter_id": tweet["id"],
                            "cashtags_tag": cash["tag"]
                        }
            
                        twitter_list_cashtags.append(cashtags)
        
            twitter_list.append(data)
            
    # dataframes
    df_twitter = pd.DataFrame(twitter_list)
    df_annotations = pd.DataFrame(twitter_list_annotations)
    df_cashtags = pd.DataFrame(twitter_list_cashtags)
    df_hashtags = pd.DataFrame(twitter_list_hashtags)
    df_mentions = pd.DataFrame(twitter_list_mentions)
    df_urls = pd.DataFrame(twitter_list_urls)
    df_public_metrics = pd.DataFrame(twitter_list_public_metrics)
    df_referenced_tweets = pd.DataFrame(twitter_list_referenced_tweets)
    
    # merge data
    if not df_annotations.empty:
        dataframe = df_twitter.merge(df_annotations,how="left",on="twitter_id")
    
    if not df_cashtags.empty:
        dataframe = dataframe.merge(df_cashtags,how="left",on="twitter_id")
        
    if not df_hashtags.empty:
        dataframe = dataframe.merge(df_hashtags,how="left",on="twitter_id")
        
    if not df_mentions.empty:  
        dataframe = dataframe.merge(df_mentions,how="left",on="twitter_id")
        
    if not df_urls.empty:    
        dataframe = dataframe.merge(df_urls,how="left",on="twitter_id")
        
    if not df_public_metrics.empty:    
        dataframe = dataframe.merge(df_public_metrics,how="left",on="twitter_id")
        
    if not df_referenced_tweets.empty:    
        dataframe = dataframe.merge(df_referenced_tweets,how="left",on="twitter_id")
        
    delete_tweets_file(files)   
    
    print("Files with successfully processed Tweets!")
    
    return dataframe


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
    
    texto = re.sub('\d+', '', str(texto)).replace("&gt;"," ").replace("&lt;"," ") 
    texto = re.sub(r"https?:\/\/\S+","", texto)
    texto = re.sub(r"@[A-Za-z0-9\w]+","", texto)
    texto = re.sub(r"#[A-Za-z0-9\w]+","", texto)
    texto = re.sub('^RT ',' ',texto)
    texto = texto.translate(trantab).replace("\n"," ")
    texto = re.sub(emoj, '', texto).replace("“"," ").replace("”"," ").strip().lower()
    texto = unidecode.unidecode(texto)
    texto = ' '.join([word for word in texto.split() if word not in stop_words_br_no_accent])
    texto = ' '.join([word for word in texto.split() if word.isalnum()])
    
    return " ".join(texto.split())

def word_frequency(sentence):
    
    sentence =" ".join(sentence)
    new_tokens = word_tokenize(sentence)
    new_tokens = [t.lower() for t in new_tokens]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    counted = Counter(new_tokens)
    word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)

    return word_freq

def show_topics(vectorizer,lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


def get_sentiment_flair(row,classifier,tipo="text"):

    # print(row.name)    

    sentence = Sentence(row.text_english)
    
    classifier.predict(sentence)
    
    if tipo != "text":
        
        return str(sentence.labels[0]).split()[1].replace("(","").replace(")","")

    else:

        return str(sentence.labels[0]).split()[0]


def get_sentiment_textblob(row):

    traducao = TextBlob(str(row)).translate(to='en')

    print( traducao.sentiment.polarity, traducao.sentiment.subjectivity)

    # traducao = TextBlob(str(texto.translate(to='en')))

    # print(traducao.sentiment)
    
    # return translate
    # traducao.sentiment_assessments
    # return [translate.sentiment.polarity, translate.sentiment.subjectivity,[val[0][0] for val in translate.sentiment_assessments.assessments]]


def get_sentiment(row, token_obj, model_obj):
    
    print(row.name)
    
    tokens = token_obj.encode(row.text_english, return_tensors='pt')
    
    result = model_obj(tokens)
          
    result.logits
    
    return int(torch.argmax(result.logits))+1


def get_bert_model():
    
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    
    return {
        "tokenizer": tokenizer,
        "model": model
    }
    
# https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f


def chart_tweets(df, params):

    x = df[params["x"]].values
    y = df[params["y"]].values
    
    total = df[params["y"]].sum()
    
    plt.figure(figsize=(10,8))
    # plt.xticks(rotation=45)
    ax = sns.barplot(x = y, y = x, data = df,palette="deep")
    ax.set_ylabel(params["y_title"], fontsize=16)
    ax.set_xlabel(params["x_title"], fontsize=16)
    
    for p in ax.patches:
        width = p.get_width()    # get bar length
        ax.text(width + 1,       # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2, # get Y coordinate + X coordinate / 2
                '{:1.2f}%'.format( (width/total)*100 ), # set variable to display, 2 decimals
                ha = 'left',   # horizontal alignment
                va = 'center')  # vertical alignment
        
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*0.01
    ymargin = (ylim[1]-ylim[0])*0.01

    ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)
    
    xlabels = ['{:,.0f}%'.format(x*100) for x in ax.get_xticks()/total]
    ax.set_xticklabels(xlabels)

    plt.text(xlim[1]+xmargin-1, (ylim[0]-ymargin)-0.6, f"Total {total:,} tweets", horizontalalignment='right', size='20', color='gray', weight='semibold')
    
    plt.title(params["title"], fontsize=16)
    plt.show()
    

def chart_tweets_trigrams(df, params):

    x = [','.join(token) for token in df[params["x"]].values]
    y = list(df[params["y"]].values)
    
    total = df[params["y"]].sum()
    
    plt.figure(figsize=(12,8))
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    # plt.rcParams["axes.labelsize"] = 10
    
    # plt.xticks(rotation=45)
    ax = sns.barplot(x = y, y = x, data = df,palette="deep")
    ax.set_ylabel(params["y_title"], fontsize=12)
    ax.set_xlabel(params["x_title"], fontsize=12)
    ax.tick_params(labelsize=12)

    for p in ax.patches:
        width = p.get_width()    # get bar length
        ax.text(width + 1,       # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2, # get Y coordinate + X coordinate / 2
                '{:1.2f}%'.format( (width/total)*100 ), # set variable to display, 2 decimals
                ha = 'left',   # horizontal alignment
                va = 'center')  # vertical alignment
        
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*0.01
    ymargin = (ylim[1]-ylim[0])*0.01

    ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)
    
    xlabels = ['{:,.0f}%'.format(x*100) for x in ax.get_xticks()/total]
    ax.set_xticklabels(xlabels)

    plt.text(xlim[1]+xmargin-1, (ylim[0]-ymargin)-0.6, f"Total {total:,} tweets", horizontalalignment='right', size='20', color='gray', weight='semibold')
    
    plt.title(params["title"], fontsize=14)
    plt.show()


def chart_tweets_sentiment(df,params):
    
    df = df[df["query"]==params["candidato"]]
    df = df.sort_values(by=['qtde'], ascending=False)

    x = df[params["x"]].values
    y = df[params["y"]].values
    
    total = df[params["y"]].sum()
    
    plt.figure(figsize=(10,6))
    #plt.xticks(rotation=45)
    ax = sns.barplot(x = y, y = x, data = df,palette="deep")
    ax.set_ylabel(params["y_title"], fontsize=16)
    ax.set_xlabel(params["x_title"], fontsize=16)
    
    for p in ax.patches:
        width = p.get_width()    # get bar length
        ax.text(width + 1,       # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2, # get Y coordinate + X coordinate / 2
                '{:1.2f}%'.format( (width/total)*100 ), # set variable to display, 2 decimals
                ha = 'left',   # horizontal alignment
                va = 'center')  # vertical alignment
        
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*0.01
    ymargin = (ylim[1]-ylim[0])*0.01
    
    print(ylim[0]-ymargin)
    
    ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)

    xlabels = ['{:,.0f}%'.format(x*100) for x in ax.get_xticks()/total]
    ax.set_xticklabels(xlabels)

    plt.text(xlim[1]+xmargin-1, (ylim[0]-ymargin)-0.3, f"Total {total} tweets", horizontalalignment='right', size='20', color='gray', weight='semibold')

    plt.title(params["title"], fontsize=16)
    plt.show()
    
    

def stemming(texto):
    stem = nltk.stem.SnowballStemmer('portuguese')
    words = texto.split() 
    stemmed_words = [stem.stem(word) for word in words]
    return " ".join(stemmed_words)

def get_stopwords():
    
    stop_words = list(set(stopwords.words("portuguese")))
    no_stop_words = ["estou","quando","qual","é","com","de","não"]
    stop_words = [item for item in stop_words if item not in no_stop_words]

    return stop_words  
    

def get_tokens(conversas_lemmatized):
    
    conversas_tokens = [item for items in conversas_lemmatized for item in items]

    return conversas_tokens


def get_qui_bigram(bigramas,buscaBigramas):
    
   QuiTabBigramas = pd.DataFrame(list(buscaBigramas.score_ngrams(bigramas.chi_sq)),
       columns = ["Bigrama","Qui"]
   ).sort_values(by = "Qui", ascending=False)
   
   return QuiTabBigramas
    
def get_qui_trigram(trigramas,buscaTrigramas):

   QuiTabTrigramas = pd.DataFrame(list(buscaTrigramas.score_ngrams(trigramas.chi_sq)),
       columns = ["Trigrama","Qui"]
   ).sort_values(by = "Qui", ascending=False)

   return QuiTabTrigramas


def get_pmi_tabfreq_bigram(bigramas,buscaBigramas,tam):
    
    buscaBigramas.apply_freq_filter(tam)
    PMITabBigramas = pd.DataFrame(list(buscaBigramas.score_ngrams(bigramas.pmi)), columns = ["Bigrama","PMI"]).sort_values(by="PMI", ascending = False)
    
    return PMITabBigramas


def get_pmi_tabfreq_trigram(trigramas,buscaTrigramas,tam):
    
    buscaTrigramas.apply_freq_filter(tam)
    PMITabTrigramas = pd.DataFrame(list(buscaTrigramas.score_ngrams(trigramas.pmi)), columns = ["Trigrama","PMI"]).sort_values(by="PMI", ascending = False)
    
    return PMITabTrigramas


def get_test_t_bigram(bigramas,buscaBigramas):
    
   TestetTabBigramas = pd.DataFrame(list(buscaBigramas.score_ngrams(bigramas.student_t)),
       columns = ["Bigrama","Teste-t"]
   ).sort_values(by = "Teste-t", ascending=False)
   
   bigramas_t_filtrados = TestetTabBigramas[TestetTabBigramas.Bigrama.map(lambda x: filtra_tipo_token_bigrama(x))]
   
   return bigramas_t_filtrados
    
def get_test_t_trigram(trigramas,buscaTrigramas):    
    
   TestetTabTrigramas = pd.DataFrame(list(buscaTrigramas.score_ngrams(trigramas.student_t)),
       columns = ["Trigrama","Teste-t"]
   ).sort_values(by = "Teste-t", ascending=False)
   trigramas_t_filtrados = TestetTabTrigramas[TestetTabTrigramas.Trigrama.map(lambda x: filtra_tipo_token_bigrama(x))]

   return trigramas_t_filtrados


def get_bigramas_freqtab(conversas_tokens):
    
    bigramas = nltk.collocations.BigramAssocMeasures()
    buscaBigramas = nltk.collocations.BigramCollocationFinder.from_words(conversas_tokens)
    
    bigrama_freq = buscaBigramas.ngram_fd.items()
    FreqTabBigramas = pd.DataFrame(list(bigrama_freq),columns = ["Bigrama","Freq"]).sort_values(by = "Freq", ascending = False)

    bigramas_filtrados = FreqTabBigramas[FreqTabBigramas.Bigrama.map(lambda x: filtra_tipo_token_bigrama(x))]
    
    return bigramas, buscaBigramas, bigramas_filtrados


def get_bigramas_freqtab_v2(conversas_tokens):
    
    buscaBigramas = nltk.collocations.BigramCollocationFinder.from_words(conversas_tokens)
    
    bigrama_freq = buscaBigramas.ngram_fd.items()
    FreqTabBigramas = pd.DataFrame(list(bigrama_freq),columns = ["Bigrama","Freq"]).sort_values(by = "Freq", ascending = False)

    bigramas_filtrados = FreqTabBigramas[FreqTabBigramas.Bigrama.map(lambda x: filtra_tipo_token_bigrama(x))]
    
    return bigramas_filtrados


def get_trigramas_freqtab_v2(conversas_tokens):

    buscaTrigramas = nltk.collocations.TrigramCollocationFinder.from_words(conversas_tokens)

    trigrama_freq = buscaTrigramas.ngram_fd.items()
    FreqTabTrigramas = pd.DataFrame(list(trigrama_freq), columns = ["Trigrama","Freq"]).sort_values(by = "Freq", ascending = False)

    trigramas_filtrados = FreqTabTrigramas[FreqTabTrigramas.Trigrama.map(lambda x: filtra_tipo_token_trigrama(x))]
    
    return trigramas_filtrados


def get_trigramas_freqtab(conversas_tokens):

    trigramas = nltk.collocations.TrigramAssocMeasures()    
    buscaTrigramas = nltk.collocations.TrigramCollocationFinder.from_words(conversas_tokens)

    trigrama_freq = buscaTrigramas.ngram_fd.items()
    FreqTabTrigramas = pd.DataFrame(list(trigrama_freq), columns = ["Trigrama","Freq"]).sort_values(by = "Freq", ascending = False)

    trigramas_filtrados = FreqTabTrigramas[FreqTabTrigramas.Trigrama.map(lambda x: filtra_tipo_token_trigrama(x))]
    
    return trigramas, buscaTrigramas, trigramas_filtrados


def filtra_tipo_token_trigrama(ngram):
    
    stop_words = get_stopwords()
    
    if "-pron-" in ngram or "t" in ngram:
        return False
    
    for word in ngram:
        if word in stop_words or word.isspace():
            return False
    
    first_type = ("JJ","JJR","JJS","NN","NNS","NNP","NNPS")
    
    second_type = ("NN","NNS","NNP","NNPS")
    
    tags = nltk.pos_tag(ngram)
    
    if tags[0][1] in first_type and tags[2][1] in second_type:
        return True
    else:
        return False  


def filtra_tipo_token_bigrama(ngram):
    
    stop_words = get_stopwords()
    
    if "-pron-" in ngram or "t" in ngram:
        return False
    
    for word in ngram:
        if word in stop_words or word.isspace():
            return False
    
    acceptable_types = ("JJ","JJR","JJS","NN","NNS","NNP","NNPS")
    
    second_type = ("NN","NNS","NNP","NNPS")
    
    tags = nltk.pos_tag(ngram)
    
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False
    
    
# funcao nuvens de palavras
def nuvem_palavras(palavras):
    todas_palavras = ' '.join([texto for texto in palavras])
    
    nuvem_palvras = WordCloud(width= 800, height= 500,
                                  max_font_size = 100,
                                  collocations = False).generate(todas_palavras)
    plt.figure(figsize=(10,8))
    plt.imshow(nuvem_palvras, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud.png', dpi=300)
    plt.show()
    
    
def get_analysis_trigram_dataframe(dataframe, token_column, num_trigram, candidate):
    
    df_qui = pd.DataFrame()
    df_pmi = pd.DataFrame()
    df_t = pd.DataFrame()
    
    trigramas = nltk.collocations.TrigramAssocMeasures()    
    for idx, item in dataframe[token_column].items():

        buscaTrigramas = nltk.collocations.TrigramCollocationFinder.from_words(item)

        # qui-quadrado
        QuiTabTrigramas = pd.DataFrame(list(buscaTrigramas.score_ngrams(trigramas.chi_sq)),
            columns = ["Trigrama","Qui"]
        ).sort_values(by = "Qui", ascending=False)
    
        # t-student
        TestetTabTrigramas = pd.DataFrame(list(buscaTrigramas.score_ngrams(trigramas.student_t)),
            columns = ["Trigrama","Teste-t"]
        ).sort_values(by = "Teste-t", ascending=False)
        
        # pmi
        PMITabTrigramas = pd.DataFrame(
            list(buscaTrigramas.score_ngrams(trigramas.pmi)), 
            columns = ["Trigrama","PMI"]
        ).sort_values(by="PMI", ascending = False)
    
        df_qui = pd.concat([df_qui,QuiTabTrigramas])
        df_pmi = pd.concat([df_pmi,PMITabTrigramas])
        df_t = pd.concat([df_t,TestetTabTrigramas])


    df_res_qui = df_qui.groupby(["Trigrama"]).agg({
        "Qui":lambda x: round(x.sum())
    }).reset_index().sort_values(by = "Qui", ascending=False)
    df_res_qui.columns = ["trigrama","score"]
    df_res_qui["trigrama"] = df_res_qui["trigrama"].apply(lambda x: list(x) if len(set(list(x))) > 1 else 1 )
    df_res_qui = df_res_qui[df_res_qui["trigrama"]!=1]
    df_res_qui["stats"] = "Qui"
    df_res_qui["candidate"] = candidate
    df_res_qui = df_res_qui.head(num_trigram)
    
    df_res_pmi = df_pmi.groupby(["Trigrama"]).agg({
        "PMI":lambda x: round(x.sum())
    }).reset_index().sort_values(by = "PMI", ascending=False)
    df_res_pmi.columns = ["trigrama","score"]
    df_res_pmi["trigrama"] = df_res_pmi["trigrama"].apply(lambda x: list(x) if len(set(list(x))) > 1 else 1 )
    df_res_pmi = df_res_pmi[df_res_pmi["trigrama"]!=1]
    df_res_pmi["stats"] = "PMI"
    df_res_pmi["candidate"] = candidate
    df_res_pmi = df_res_pmi.head(num_trigram)

    df_res_t = df_t.groupby(["Trigrama"]).agg({
        "Teste-t":lambda x: round(x.sum())
    }).reset_index().sort_values(by = "Teste-t", ascending=False)
    df_res_t.columns = ["trigrama","score"]
    df_res_t["trigrama"] = df_res_t["trigrama"].apply(lambda x: list(x) if len(set(list(x))) > 1 else 1 )
    df_res_t = df_res_t[df_res_t["trigrama"]!=1]
    df_res_t["stats"] = "Teste-t"
    df_res_t["candidate"] = candidate
    df_res_t = df_res_t.head(num_trigram)

    return pd.concat([df_res_qui, df_res_pmi, df_res_t])


def get_dataframe_tokens(dataframe, sentiment_analysis_model, candidate):
    
    # stemming em portuges
    stemmer = RSLPStemmer()

    # verifica quais dados de analise de sentimentos vai usar com base no modelo aplicado
    if sentiment_analysis_model == "flair":
        
        df = dataframe[
            (dataframe["query"]==candidate) & 
            (dataframe["flair_sentiment_text"]=="POSITIVE") & 
            (dataframe["flair_sentiment_accuracy"]>=0.90) 
        ]
        
    elif sentiment_analysis_model == "bert":
    
        df = dataframe[
            (dataframe["query"]==candidate) &
            (dataframe["bert_sentiment_level_text"]=="VERY_POSITIVE")
        ]
    
    stopwords_ = stopwords.words("portuguese")
    
    stopwords_.extend(["q","vc","tá","ex","ai","lá","né","vcs","pq","p","dá","x","º","kkkkk","irrrrrrrrrrrrrriiiiiiiiiiiiiiii",
                       "bom","boa","dia","noite","tarde","ficar","ligado","tudo", "aí","cara","acho","kkkk"])
    
    candidate_list = []
    temp = [candidate_list.extend(word.split()) for word in list(dataframe["query"].unique()) if word not in [candidate]]
    
    stopwords_.extend(candidate_list)
    
    df['text_clean_lemma'] = df['text_clean'].apply(lambda frase: 
        [token.lemma_ for token in nlp(frase) if ((token.pos_ in ["ADV","NOUN"]) & (token.text not in stopwords_)) ] 
    )
        
    df['text_clean_stemm'] = df['text_clean'].apply(lambda frase: 
        [stemmer.stem(word) for word in frase.split() if word not in stopwords_]
    )
    
    df['text_clean_tokens'] = df['text_clean'].apply(lambda frase: 
        [word for word in frase.split() if word not in stopwords_]
    )
        
    return df[[
        "twitter_id",
        "have_retweet",
        "have_like",
        "retweet_count",
        "like_count",
        "author_id",
        "text_clean_lemma",
        "text_clean_stemm",
        "text_clean_tokens"
    ]]


def get_topicos(dataframe, token_column, num_features, num_topicos, num_top_words):
    
    documents = [' '.join([w for w in frase]) for frase in dataframe[token_column]]

    # LDA
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features)
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Run LDA
    t2 = time()
    lda = LatentDirichletAllocation(
        n_components = num_topicos, 
        max_iter=5, 
        learning_method='online', 
        learning_offset=50.,
        random_state=0
    ).fit(tf)
    print("done in %0.3fs." % (time() - t2))

    topic_list = []
    for topic_idx, topic in enumerate(lda.components_):
            
        topic_dic = {}
        topic_dic["topic"] = topic_idx+1
        topic_dic["text"] = " ".join([tf_feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]])
        topic_list.append(topic_dic)
        
    return pd.DataFrame(topic_list)


def get_tweets_trigram(trigrama, twitter_id_list, tokens_list):
    
    tw = []
    for idx, item in enumerate(tokens_list):
        
        if set(trigrama).issubset(item):
            tw.append(twitter_id_list[idx])
        
    return len(list(set(tw)))

def get_tweets_topicos(topicos, twitter_id_list, tokens_list):
    
    tw = []
    for idx, item in enumerate(tokens_list):
        
        if set(topicos.split()).issubset(item):
            tw.append(twitter_id_list[idx])

    return len(list(set(tw)))



def get_analysis_trigram(dataframe, model, candidate, num_trigram=50, chart_limit_terms=10, model_stats="Teste-t", wordcloud=False):
    
    start = time()

    df_tokens = get_dataframe_tokens(dataframe, model, candidate)
    
    df_trigrams_tokens = get_analysis_trigram_dataframe(
        dataframe = df_tokens, 
        token_column = "text_clean_tokens", 
        num_trigram = num_trigram, 
        candidate = candidate)
    
    df_trigrams_tokens["twitter_count"] = df_trigrams_tokens["trigrama"].apply(lambda x: get_tweets_trigram(
        trigrama=x,
        twitter_id_list=df_tokens["twitter_id"].values,
        tokens_list=df_tokens["text_clean_tokens"].values) 
    )
    
    df_temp = df_trigrams_tokens[
        df_trigrams_tokens["stats"]==model_stats
    ][["trigrama","twitter_count"]].sort_values(by = "twitter_count", ascending=False)
    
    chart_tweets_trigrams(
        df = df_temp.head(chart_limit_terms),
        params = {
            "x": "trigrama",
            "y": "twitter_count",
            "annotation_w":1e1,
            "annotation_h":1,
            "x_title": "% Tweets",
            "y_title": "Termos mais relevantes",
            "title":f"TOP {len(df_temp.head(chart_limit_terms))} termos mais relevantes no Twitter - Candidato {candidate.upper()}"
        }
    )
    
    if wordcloud:
        
        nuvem_palavras([' '.join(list(word)) for word in df_trigrams_tokens[df_trigrams_tokens["stats"]=="Teste-t"]["trigrama"]])
    
    end = time()
    print(f"Finished: {end-start}")
    
    
def get_analysis_topics(
        dataframe, 
        model,
        candidate, 
        token_column="text_clean_tokens",
        num_features=1000,
        num_topicos=100,
        num_top_words=3,
        chart_limit_terms=10, 
        wordcloud=False):  
    
    df_tokens = get_dataframe_tokens(dataframe, model, candidate)
    
    df_topicos = get_topicos(
        dataframe = df_tokens,
        token_column = token_column,
        num_features = num_features,
        num_topicos = num_topicos,
        num_top_words = num_top_words
    )
    
    df_topicos["twitter_count"]=df_topicos["text"].apply(lambda x: get_tweets_topicos(
        topicos=x,
        twitter_id_list=df_tokens["twitter_id"].values,
        tokens_list=df_tokens[token_column].values) 
    )
    df_topicos.sort_values(by = "twitter_count", ascending=False, inplace=True)
    df_topicos["text"] = df_topicos["text"].apply(lambda x: x.split())
    df_topicos = df_topicos[df_topicos["twitter_count"]>0]
    
    chart_tweets_trigrams(
        df = df_topicos.head(chart_limit_terms),
        params = {
            "x": "text",
            "y": "twitter_count",
            "annotation_w":1e1,
            "annotation_h":1,
            "x_title": "% Tweets",
            "y_title": "Tópicos mais relevantes",
            "title":f"TOP {len(df_topicos.head(chart_limit_terms))} tópicos mais relevantes no Twitter - Candidato {candidate.upper()}"
        }
    )

    if wordcloud:
        
        nuvem_palavras([' '.join(list(word)) for word in df_topicos["text"]])












