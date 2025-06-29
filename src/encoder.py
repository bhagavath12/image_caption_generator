# encoder.py

import tensorflow as tf
from tensorflow.keras import layers

class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)  # shape: (batch_size, embedding_dim)
        x = tf.nn.relu(x)
        return x
