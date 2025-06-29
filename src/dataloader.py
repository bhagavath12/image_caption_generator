import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
CAPTIONS_FILE = os.path.join("data", "captions_clean.csv")
FEATURES_DIR = os.path.join("data", "features")
TOKENIZER_PATH = os.path.join("data", "tokenizer.pkl")

# Parameters
MAX_LENGTH = 35       # Maximum caption length
BATCH_SIZE = 64       # Batch size
BUFFER_SIZE = 1000    # Shuffle buffer

def load_tokenizer():
    """Load the tokenizer from disk."""
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def map_func(img_name, cap):
    """Load image feature .npy and pair with caption sequence."""
    # Decode bytes to string if needed
    if isinstance(img_name, bytes):
        img_name = img_name.decode("utf-8")

    # Strip '.jpg' and any '#0' if present
    base_name = img_name.split('.')[0]
    base_name = base_name.split('#')[0]

    # Form full path to feature file
    feature_path = os.path.join(FEATURES_DIR, base_name + ".npy")

    # Check if file exists
    if not os.path.exists(feature_path):
        print(f"⚠️ Missing file: {feature_path}. Returning zero tensor.")
        return np.zeros((2048,), dtype=np.float32), cap

    # Load the feature tensor
    img_tensor = np.load(feature_path).astype(np.float32)
    return img_tensor, cap

def create_dataset():
    """Create TensorFlow dataset from captions and image features."""
    df = pd.read_csv(CAPTIONS_FILE)
    tokenizer = load_tokenizer()

    print("🔤 Tokenizing captions...")
    cap_seqs = tokenizer.texts_to_sequences(df['caption'].values)
    cap_seqs = pad_sequences(cap_seqs, maxlen=MAX_LENGTH, padding='post')

    print("🖼️ Preparing image names...")
    img_names = df['image_id'].values  # E.g., 1000268201_693b08cb0e.jpg

    print(f"📦 Dataset size: {len(img_names)}")

    # Build TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_names, cap_seqs))

    def tf_map_func(img, cap):
        return tf.numpy_function(map_func, [img, cap], [tf.float32, tf.int32])

    dataset = dataset.map(tf_map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset, tokenizer
