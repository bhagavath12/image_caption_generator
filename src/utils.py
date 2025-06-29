# src/utils.py

import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_caption(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return f"<start> {text.strip()} <end>"

def build_tokenizer(captions, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer

def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def tokenize_and_pad(captions, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(captions)
    return pad_sequences(sequences, maxlen=max_len, padding="post")
