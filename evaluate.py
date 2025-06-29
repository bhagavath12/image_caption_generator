import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from nltk.translate.bleu_score import corpus_bleu

from src.encoder import CNN_Encoder
from src.decoder import RNN_Decoder
from src.config import MAX_LENGTH, FEATURES_DIR, TOKENIZER_PATH
from src.utils import load_image_features

# Load tokenizer
def load_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as f:
        return pickle.load(f)

# Load model components
def load_models(embedding_dim=256, units=512, vocab_size=None):
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    return encoder, decoder

# Evaluate function
def evaluate_model(captions_file, encoder, decoder, tokenizer):
    df = pd.read_csv(captions_file)
    actual, predicted = [], []

    for idx, row in df.iterrows():
        image_id = row['image_id'].split('.')[0]
        feature_path = os.path.join(FEATURES_DIR, image_id + ".npy")

        if not os.path.exists(feature_path):
            continue

        img_tensor = np.load(feature_path)
        img_tensor = tf.expand_dims(img_tensor, 0)
        hidden = decoder.reset_state(batch_size=1)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        features = encoder(img_tensor)

        for _ in range(MAX_LENGTH):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            predicted_id = tf.argmax(predictions[0]).numpy()
            word = tokenizer.index_word.get(predicted_id, '')
            if word == '<end>' or word == '':
                break
            result.append(word)
            dec_input = tf.expand_dims([predicted_id], 0)

        actual_caption = row['caption'].replace('<start>', '').replace('<end>', '').strip().split()
        actual.append([actual_caption])
        predicted.append(result)

        if idx % 100 == 0:
            print(f"Evaluated {idx} captions...")

    bleu_score = corpus_bleu(actual, predicted)
    print(f"\n BLEU Score: {bleu_score:.4f}")

if __name__ == '__main__':
    print(" Loading tokenizer...")
    tokenizer = load_tokenizer()

    print(" Loading models...")
    encoder, decoder = load_models(vocab_size=len(tokenizer.word_index) + 1)

    print(" Running BLEU evaluation...")
    evaluate_model("data/captions_clean.csv", encoder, decoder, tokenizer)
