# decoder.py

import tensorflow as tf
from tensorflow.keras import layers

class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units

        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform')
        self.fc1 = layers.Dense(units)
        self.fc2 = layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        # x: input word token
        # features: image features from encoder
        # hidden: previous hidden state of LSTM

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(features, 1), x], axis=-1)

        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        x = self.fc1(output)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x, state_h, state_c

    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
