import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image

from src.encoder import CNN_Encoder
from src.decoder import RNN_Decoder
from src.utils import load_image, extract_image_features

# --- Configuration ---
IMAGE_PATH = "test.jpg"  # Change this to your custom image
MAX_LENGTH = 35
FEATURE_SHAPE = 2048
TOKENIZER_PATH = "data/tokenizer.pkl"
CHECKPOINT_PATH = "checkpoints/train"

# --- Load Tokenizer ---
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# --- Load Encoder and Decoder ---
encoder = CNN_Encoder(FEATURE_SHAPE)
decoder = RNN_Decoder(embedding_dim=256, units=512, vocab_size=len(tokenizer.word_index) + 1)

checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH)).expect_partial()
print("✅ Loaded model from checkpoint")

# --- Preprocess Image ---
def load_and_extract(img_path):
    img_tensor = extract_image_features(img_path)  # You should have this in utils.py
    return tf.expand_dims(img_tensor, 0)  # Add batch dimension

# --- Generate Caption ---
def generate_caption(image_path):
    attention_plot = np.zeros((MAX_LENGTH, 64))
    hidden = decoder.reset_state(batch_size=1)

    features = load_and_extract(image_path)
    enc_out = encoder(features)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(MAX_LENGTH):
        predictions, hidden, _ = decoder(dec_input, enc_out, hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = tokenizer.index_word.get(predicted_id, '<unk>')

        if predicted_word == '<end>':
            break

        result.append(predicted_word)
        dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)

# --- Run ---
if __name__ == "__main__":
    caption = generate_caption(IMAGE_PATH)
    print(f" Image: {IMAGE_PATH}")
    print(f" Caption: {caption}")
