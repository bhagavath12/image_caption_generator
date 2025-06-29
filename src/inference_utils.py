import tensorflow as tf

def generate_caption_greedy(image_features, tokenizer, decoder, max_length=35):
    result = []
    input_seq = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    hidden_state = decoder.reset_state(batch_size=1)

    for _ in range(max_length):
        predictions, hidden_state, _ = decoder(input_seq, image_features, hidden_state)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = tokenizer.index_word.get(predicted_id, '')

        if word == '<end>':
            break
        result.append(word)
        input_seq = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)
