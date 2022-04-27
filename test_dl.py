import numpy as np 
import unidecode
import pandas as pd 
from nltk.stem.rslp import RSLPStemmer
import re
import string
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

lemmatizer = RSLPStemmer()

dataframe = pd.read_csv("/home/daholive/Documents/twitter_ellection_brazil/datasource/TweetsWithTheme.csv", sep=",")

dataframe["sentiment"] = dataframe["sentiment"].map({
    "Negativo": 0,
    "Positivo": 1    
})


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


data = np.array(dataframe["tweet_text"].map(clean_text).values)

y = np.array(dataframe['sentiment'])
labels = tf.keras.utils.to_categorical(y, 2, dtype="float32")

max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)
print(labels)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(tweets,labels, random_state=0)
print(len(X_train),len(X_test),len(y_train),len(y_test))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)


# Single LSTM layer model
model1 = Sequential()
model1.add(tf.keras.layers.Embedding(max_words, 20))
model1.add(tf.keras.layers.LSTM(15,dropout=0.5))
model1.add(tf.keras.layers.Dense(2,activation='softmax'))
# model1.save('models/best_model1.hdf5')

model1.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint1 = ModelCheckpoint(
    "models/best_model1.hdf5", 
    monitor='val_accuracy', 
    verbose=1,
    save_best_only=True, 
    mode='auto', 
    save_freq=1,
    save_weights_only=False)

history = model1.fit(
    X_train, 
    y_train, 
    epochs=70,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint1])





import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Bidirectional LTSM model
# from tensorflow.keras.optimizers import SGD
# opt = SGD(lr=0.01)

# from keras.optimizers import SGD
# opt = SGD(lr=0.01)
# model.compile(loss = "categorical_crossentropy", optimizer = opt)

model2 = Sequential()
model2.add(layers.Embedding(max_words, 40, input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model2.add(layers.Dense(2,activation='softmax'))
model2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

# model2.summary()
# model2.save('models/best_model2.hdf5')
#'rmsprop'

#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint2 = ModelCheckpoint(
    filepath="models/best_model_v2.hdf5", 
    monitor='accuracy', 
    verbose=1,
    save_best_only=True, 
    mode='auto', 
    save_freq=1,
    save_weights_only=False)

history = model2.fit(
    X_train, 
    y_train, 
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint2])




from pathlib import Path
# Save neural network structure
model_structure = model2.to_json()
f = Path("models/model2.json")
f.write_text(model_structure)
print('done')

# Save neural network's trained weights
model2.save_weights("models/model2_weights.h5")
print('done')

# or you can save the full model via:
model2.save('models/model2.h5')

#delete your model in memory
del model2

#Know to load your model use:
my_new_model = tf.keras.models.load_model("models/model2.h5")


#compile my_new_model:
my_new_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])




# Best model validation
#Let's load the best model obtained during training
best_model = tf.keras.models.load_model("models/best_model2.hdf5")

test_loss, test_acc = my_new_model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ',test_acc)

predictions = best_model.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
matrix = confusion_matrix(y_test.argmax(axis=1), np.around(predictions, decimals=0).argmax(axis=1))

import seaborn as sns
conf_matrix = pd.DataFrame(matrix, index = [0,1],columns = [0,1])
#Normalizing
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize = (15,15))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})


# TEST
sentiment = [0,1]

token = clean_text('Você é um amigo mal humorado')

sequence = tokenizer.texts_to_sequences([token])
test = pad_sequences(sequence, maxlen=max_len)
sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]



# Preparing model for AWS SageMaker

# #Saving weights and tokenizer so we can reduce training time on SageMaker
# import pickle
# # serialize model to JSON
# model_json = best_model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# best_model.save_weights("model-weights.h5")
# print("Model saved")

# # saving tokenizer
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('Tokenizer saved')













