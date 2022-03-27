# Projeto Análise da Popularidade dos Candidatos a Eleição de 2022 no Brasil

## Pontos de análise:

- Total de tweets por candidato
- Media de tweets por dia por candidato

- Total de tweets com retweets por canditatos

- Total de tweets com likes por canditatos

- Assuntos mais citados dos candidatos com aplicação de modelos de análise de sentimentos:

  - **Modelo Bert**, ref.: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
  
    **Description:** This a bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in  six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5). This model is intended for direct use as a sentiment analysis model for product reviews in any of the six languages above, or for further finetuning on related sentiment analysis tasks.
    
  - **Modelo Flair**, ref.: https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f
   
    **Description:** Flair delivers state-of-the-art performance in solving NLP problems such as named entity recognition (NER), part-of-speech tagging (PoS), sense disambiguation and text classification. It’s an NLP framework built on top of PyTorch.
   

    
## Analises Textuais
    
- Top assuntos mais citados nos tweets por sentimento por candidatos  (Aplicar Trigrama)
- Top assuntos mais citados nos tweets por sentimento por candidatos  (Aplicar LDA)
- Representatividade dos tweets positivos:

 - % Tweets positivos

  - % Retweets
  - % Likes
  - % Termos mais relevantes por candidato (Trigrama)
  - % Tópicos mais relevantes por candidato (LDA)
