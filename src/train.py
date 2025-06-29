# train.py

import tensorflow as tf
import os
from encoder import Encoder
from decoder import Decoder
from dataloader import create_dataset

# Config
embedding_dim = 256
units = 512
EPOCHS = 1  # ⏱️ Start with 1 to test, then increase
CHECKPOINT_PATH = "./checkpoints"

# Load data
dataset, tokenizer = create_dataset()
vocab_size = len(tokenizer.word_index) + 1

# Init models
encoder = Encoder(embedding_dim)
decoder = Decoder(embedding_dim, units, vocab_size)

# Optimizer and loss
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# Loss function
def loss_function(real, pred):
    mask = tf.math.not_equal(real, 0)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    return tf.reduce_mean(loss_ * mask)

# Training step
@tf.function
def train_step(img_tensor, target):
    loss = 0

    # Prepare decoder input: start token
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    hidden = decoder.reset_state(batch_size=target.shape[0])

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            predictions, hidden_h, hidden_c = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions[:, 0, :])
            dec_input = tf.expand_dims(target[:, i], 1)
            hidden = [hidden_h, hidden_c]

    total_loss = loss / int(target.shape[1])
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    return total_loss

# Checkpoint manager
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=3)

# Train loop
for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss = train_step(img_tensor, target)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(f"🌀 Epoch {epoch+1} Batch {batch} Loss {batch_loss:.4f}")

    print(f"✅ Epoch {epoch+1} Loss: {total_loss / batch:.6f}")

    ckpt_manager.save()
    print("💾 Checkpoint saved.")
