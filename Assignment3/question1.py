import collections
import json
import numpy as np
import os
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM, SimpleRNN
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

def load_data(path):
    data_file = os.path.join(path)
    with open(data_file,"r") as f:
        lang_data = f.read()
    return lang_data.split('\n')


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    print(preprocess_y.shape)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    print(preprocess_y.shape)

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def get_model(shape, m_type, size):
    learning_rate = 1e-3
    input_seq = Input(shape[1:])
    rnn = m_type(64, return_sequences = True)(input_seq)
    rnn2 = m_type(64, return_sequences = True)(rnn)
    logits = TimeDistributed(Dense(size))(rnn2)
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss = sparse_categorical_crossentropy, 
                  optimizer = Adam(learning_rate), 
                  metrics = ['accuracy'])
    return model

def save_predictions(preds, engs, fres, file_name):
    with open(file_name, "w") as f:
        for eng, fre, pred in zip(engs, fres, preds):
            f.write(f"English:    {eng}\n")        
            f.write(f"French:     {fre}\n")
            f.write(f"Translated: {logits_to_text(pred, french_tokenizer)}\n\n")

if __name__ == '__main__':
    # Load English data
    english_sentences = load_data('data/small_vocab_en.txt')
    # Load French data
    french_sentences = load_data('data/small_vocab_fr.txt')
    print('Dataset Loaded')
    
    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
         preprocess(english_sentences, french_sentences)
    
    max_english_sequence_length = preproc_english_sentences.shape[1]
    max_french_sequence_length = preproc_french_sentences.shape[1]
    english_vocab_size = len(english_tokenizer.word_index)
    french_vocab_size = len(french_tokenizer.word_index)+1
    
    # Reshaping the input to work with a basic RNN
    tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))
    print(tmp_x.shape[1:])

    # SimpleRNN
    print("RNN")
    rnn_model = get_model(tmp_x.shape, SimpleRNN, french_vocab_size)
    rnn_model.summary()
    start = time.time()
    rnn_history = rnn_model.fit(tmp_x[:100000], preproc_french_sentences[:100000], batch_size=16, epochs=10, validation_split=0.2)
    rnn_history.history["training_time"] = time.time() - start
    rnn_model.save("rnn.keras")

    # LSTM
    print("LSTM")
    lstm_model = get_model(tmp_x.shape, LSTM, french_vocab_size)
    lstm_model.summary()
    start = time.time()
    lstm_history = lstm_model.fit(tmp_x[:100000], preproc_french_sentences[:100000], batch_size=16, epochs=10, validation_split=0.2)
    lstm_history.history["training_time"] = time.time() - start
    lstm_model.save("lstm.keras")

    # GRU
    print("GRU")
    gru_model = get_model(tmp_x.shape, GRU, french_vocab_size)
    gru_model.summary()
    start = time.time()
    gru_history = gru_model.fit(tmp_x[:100000], preproc_french_sentences[:100000], batch_size=16, epochs=10, validation_split=0.2)
    gru_history.history["training_time"] = time.time() - start
    gru_model.save("gru.keras")


#    rnn_model = tf.keras.models.load_model("rnn.keras")
#    prediction = rnn_model.predict(tmp_x[:10])
#    save_predictions(prediction, english_sentences, french_sentences, "rnn_samples")
#    lstm_model = tf.keras.models.load_model("lstm.keras")
#    prediction = lstm_model.predict(tmp_x[:10])
#    save_predictions(prediction, english_sentences, french_sentences, "lstm_samples")
#    gru_model = tf.keras.models.load_model("gru.keras")
#    prediction = gru_model.predict(tmp_x[:10])
#    save_predictions(prediction, english_sentences, french_sentences, "gru_samples")

    with open('RNN_history.json', 'w') as f:
            json.dump(rnn_history.history, f)
    with open('LSTM_history.json', 'w') as f:
            json.dump(lstm_history.history, f)
    with open('GRU_history.json', 'w') as f:
            json.dump(gru_history.history, f)

    rnn_results  =  rnn_model.evaluate(tmp_x[100000:], preproc_french_sentences[100000:])
    lstm_results = lstm_model.evaluate(tmp_x[100000:], preproc_french_sentences[100000:])
    gru_results  =  gru_model.evaluate(tmp_x[100000:], preproc_french_sentences[100000:])

    with open('RNN_results.json', 'w') as f:
            json.dump(rnn_results, f)
    with open('LSTM_results.json', 'w') as f:
            json.dump(lstm_results, f)
    with open('GRU_results.json', 'w') as f:
            json.dump(gru_results, f)

