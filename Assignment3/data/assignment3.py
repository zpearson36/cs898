import collections
import numpy as np
import os
import tensorflow as tf


if __name__ == "__main__":
    print("Loading Data")
    with open("data/small_vocab_en.txt", "r") as f:
        en = f.read().split("\n")
    with open("data/small_vocab_fr.txt", "r") as f:
        fr = f.read().split("\n")

    print("Creating Tokenizers")
    en_tokenizer = tf.keras.layers.TextVectorization(output_mode="int")
    fr_tokenizer = tf.keras.layers.TextVectorization(output_mode="int")

    print("Tokenizing English")
    en_tokenizer.adapt(en)
    print(en_tokenizer.get_vocabulary()[:10])
    print("Tokenizing French")
    fr_tokenizer.adapt(fr)
    print(fr_tokenizer.get_vocabulary()[:10])

    print("Vectorizing Sentence")
    print(en_tokenizer(en[0]))


