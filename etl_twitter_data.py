from datetime import datetime, timedelta
from helpers import get_recents_tweets, save_tweets_file, read_tweet_files, save_tweets_dataframe
import argparse
import os
import logging
import sys

# configuracao do parametro que sera passado por linha de comando
parser=argparse.ArgumentParser()
parser.add_argument('--interval', help='Defini o tempo de intervalo a cada request em minutos.')
parser.add_argument('--pagination', help='Defini o número de páginas do request.')
parser.add_argument('--path_dataframe', help='Defini e caminho onde será salvo os dataframes coletados')
args=parser.parse_args()

# parametro passado por linha de comando
request_interval = args.interval
page_number = args.pagination
path_dataframe = args.path_dataframe
path_log = args.path_dataframe

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('twitter.log','w+')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# horario de coleta com 4 minutos de atraso
end_time = (datetime.utcnow() - timedelta(minutes = int(request_interval))).strftime('%Y-%m-%dT%H:%M:%SZ')

precandidatos2022_list = [
    "lula",
    "bolsonaro",
    "sergio moro",
    "ciro gomes",
    "joão doria",
    "rodrigo pacheco",
    "simone tebet",
    "alessandro vieira",
    "aldo rebelo",
    "andré janones",
    "felipe d'ávila",
    "leonardo péricles"      
]

logger.info("### Started ###")

for candidatos in precandidatos2022_list: 

    # faz a coleta dos tweets com limit de 10 requests e palavra chave de consulta
    tweets_list = get_recents_tweets(logger, end_time, limit=int(page_number), query=candidatos)

    # salva os tweets coletados em arquivo
    save_tweets_file(tweets_list)

# consolida os tweets coletados em um dataframe pandas
dataframe = read_tweet_files()

# salva dataframe com dados coletados
save_tweets_dataframe(dataframe, path_dataframe)

logger.info("### Finished ###")








