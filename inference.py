import numpy as np
import tensorflow as tf
import pickle
from src.encoder import CNN_Encoder
from src.decoder import RNN_Decoder
from src.inference_utils import generate_caption_greedy
from src.utils import load_image  # Make sure you have this function

# Paths
TOKENIZER_PATH = "data/tokenizer.pkl"
FEATURES_DIR = "data/features/"
IMAGE_NAME = "1000268201_693b08cb0e"
FEATURE_PATH = FEATURES_DIR + IMAGE_NAME + ".npy"

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Load features
img_tensor = np.load(FEATURE_PATH).astype(np.float32)
img_tensor = tf.expand_dims(img_tensor, 0)

# Load models (adjust units, vocab size accordingly)
encoder = CNN_Encoder(embedding_dim=256)
decoder = RNN_Decoder(embedding_dim=256, units=512, vocab_size=len(tokenizer.word_index) + 1)

# Load weights
encoder.load_weights('checkpoints/encoder.ckpt')  # Adjust if you saved differently
decoder.load_weights('checkpoints/decoder.ckpt')

# Inference
features = encoder(img_tensor)
caption = generate_caption_greedy(features, tokenizer, decoder)
print("🖼️ Caption:", caption)
