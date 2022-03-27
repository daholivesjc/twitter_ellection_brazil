import pandas as pd
import numpy as np
import warnings
from helpers import ( 
    chart_tweets_trigrams,
    chart_tweets,
    chart_tweets_sentiment,
    nuvem_palavras,
    get_analysis_trigram_dataframe,
    get_dataframe_tokens,
    get_topicos,
    get_tweets_trigram,
    get_analysis_trigram,
    get_analysis_topics
)
import nltk
import spacy
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('rslp')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("pt_core_news_lg")

# Definindo os estilos dos gráficos
# %matplotlib inline
# plt.style.use('fivethirtyeight')

# Ignorando os avisos
warnings.filterwarnings('ignore')

# dataframe tweets
dataframe = pd.read_parquet("/home/daholive/Documents/twitter_ellection_brazil/datasource/tweets_preprocessing.parquet")

dataframe['flair_sentiment_accuracy'] = pd.to_numeric(dataframe['flair_sentiment_accuracy'],errors='coerce')

"""
Perguntas de negocios:
    - Total de tweets por candidato
    - Media de tweets por dia por candidato
    
    - Total de tweets com likes por canditatos
    - Media de tweets com likes por dia por candidato
    
    - Total de tweets com retweets por canditatos
    - Media de tweets com retweets por dia por candidato

    - Total de tweets por nivel de sentimento 
       (muito_negativo, negativo, neutro, positivo, muito_positivo) 
       
    - Total de tweets por nivel de sentimento agrupado
       (negativo, neutro, positivo) 
    
    # Analises Textuais
    
    - Assuntos mais citados nos tweets por sentimento por candidatos  (Aplicar LDA)
    - Representatividade dos tweets positivos 
    
    x % representa positivos

        x % teve retweets
        
        x % teve likes

        x % candidato a
        
          x % assunto ab
          x % assunto ac
          x % assunto ad

        x % candidato b
        
        x % candidato c

"""
###############################################################################
# Total de tweets por candidato
# incluir o periodo dos dados no grafico
df1 = dataframe.groupby(["query"]).agg({
    "twitter_id": "nunique"
}).reset_index().rename(columns={"twitter_id":"qtde"}).sort_values(by=['qtde'], ascending=False)

min_date = pd.to_datetime(dataframe["dated_at_tz"]).min().strftime("%d/%m/%Y")
max_date = pd.to_datetime(dataframe["dated_at_tz"]).max().strftime("%d/%m/%Y")

chart_tweets(
    df = df1,
    params = {
        "x": "query",
        "y": "qtde",
        "annotation_w":3e4,
        "annotation_h":8,
        "x_title": "% Tweets",
        "y_title": "candidatos",
        "title":f"Tweets por candidatos (em %) - ({min_date} - {max_date})"
    }
)


###############################################################################
# Total de tweets por dia por candidatos
df2 = dataframe.groupby(["query","dated_at_tz"]).agg({
    "twitter_id": "nunique"
}).reset_index().rename(columns={"twitter_id":"qtde","dated_at_tz": "data"}).groupby([
    "query"
]).agg({
    "qtde": lambda x: round(x.mean())
}).reset_index().sort_values(by=['qtde'], ascending=False)
    

chart_tweets(
    df = df2,
    params = {
        "x": "query",
        "y": "qtde",
        "annotation_w":5e3,
        "annotation_h":8,
        "x_title": "% Tweets",
        "y_title": "candidatos",
        "title":"Média diária de tweets por candidatos (em %)"
    }
)


###############################################################################
### ANALISE REPRESENTATIVIDADE TWITTER
###############################################################################

# total de tweets
total_tweets_unicos = dataframe["twitter_id"].nunique()


# total de tweets com sentimento positivo e 90% acuracia - MODELO SENTIMENT ANALYSIS FLAIR
total_tweets_unicos_positivos = dataframe[
    (dataframe["flair_sentiment_text"]=="POSITIVE") & 
    (dataframe["flair_sentiment_accuracy"]>=0.90) 
]["twitter_id"].nunique()

# total de autores unicos dos tweets
total_tweets_autores_unicos_positivos = dataframe[
    (dataframe["flair_sentiment_text"]=="POSITIVE") & 
    (dataframe["flair_sentiment_accuracy"]>=0.90) 
]["author_id"].nunique()

# percentual tweets positivos
percentual_tweets_positivos = round(total_tweets_unicos_positivos / total_tweets_unicos, 2)

# percentual de tweets positivos por candidato
dataframe[
    (dataframe["flair_sentiment_text"]=="POSITIVE") & 
    (dataframe["flair_sentiment_accuracy"]>=0.90) 
].groupby(["query"]).agg({
    "twitter_id": lambda x: round(len(x)/total_tweets_unicos_positivos,4)*100,
    "have_retweet": lambda x: round(sum(x)/total_tweets_unicos_positivos,4)*100,
    "have_like": lambda x: round(sum(x)/total_tweets_unicos_positivos,4)*100,
    "author_id": lambda x: round(len(np.unique(x))/total_tweets_autores_unicos_positivos,4)*100
}).reset_index().rename(
    columns={
        "twitter_id":"perc_tweets",
        "have_retweet": "perc_retweets",
        "have_like": "perc_like",
        "author_id": "perc_active_users",
    }
).sort_values(by = "perc_tweets", ascending=False)

 



###############################################################################
### ANALISE TEXTO TWITTER
###############################################################################
    
# dataframe["query"].unique()

get_analysis_trigram(
    dataframe=dataframe, 
    model="flair", 
    candidate="lula", 
    num_trigram=50, 
    chart_limit_terms=20,
    model_stats="Teste-t",
    wordcloud=True
)

get_analysis_topics(
    dataframe=dataframe, 
    model="flair", 
    candidate="lula", 
    token_column="text_clean_tokens",
    num_features=1000,
    num_topicos=100,
    num_top_words=3,
    chart_limit_terms=20, 
    wordcloud=True
)








