import tensorflow as tf

def tokenize_data(data):
    tokenized_text = tf.keras.preprocessing.text.text_to_word_sequence(input_text=data)

    vocab = sorted(set(tokenized_text))
    tokenized_text_size = len(tokenized_text)

    return (vocab, tokenized_text_size, tokenized_text)