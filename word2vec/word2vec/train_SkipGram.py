from enum import unique
import tensorflow as tf

tf.random.set_seed(42)

from pyimagesearch import config
from pyimagesearch.create_vocabulary import tokenize_data
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
import numpy as np
import os

print("[INFO] reading the data from disk ...")
with open('data.txt') as f:
    lines = f.readlines()
text_data = "".join(lines)

(vocab, tokenize_text_size, tokenized_text) = tokenize_data(text_data)

vocab_to_index = {
    unique_word:index for (index, unique_word) in enumerate(vocab)
}
index_to_vocab = np.array(vocab)

text_as_int = np.array([vocab_to_index[word] for word in tokenized_text])

context_vector_matrix = tf.Variable(
    np.random.rand(tokenize_text_size, config.EMBEDDING_SIZE)
)

center_vector_matrix = tf.Variable(
    np.random.rand(tokenize_text_size, config.EMBEDDING_SIZE)
)

optimizer = tf.optimizers.Adam()
loss_list = []

print("[INFO] starting skipgram training ...")
for iter in tqdm(range(config.ITERATIONS)):
    loss_per_epoch = 0

    for start in range(tokenize_text_size - config.WINDOW_SIZE):
        indices = text_as_int[start:start + config.WINDOW_SIZE]
    
    with tf.GradientTape() as tape:
        loss = 0

        center_vector = center_vector_matrix[indices[config.WINDOW_SIZE // 2], :]
        output = tf.matmul(
            context_vector_matrix, tf.expand_dims(center_vector, 1)
        )

        softmax_output = tf.nn.softmax(output, axis=0)

        for (count, index) in enumerate(indices):
            if count != config.WINDOW_SIZE // 2:
                loss += softmax_output[index]

            logloss = -tf.math.log(loss)

        loss_per_epoch += logloss.numpy()
        grad = tape.gradient(
            logloss, [context_vector_matrix, center_vector_matrix]
        )
        optimizer.apply_gradients(
            zip(grad, [context_vector_matrix, center_vector_matrix])
        )
    loss_list.append(loss_per_epoch)

if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)

print("[INFO] plotting loss ...")
plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(config.SKIPGRAM_LOSS)


# print(center_vector_matrix.numpy())

tsneEmbed = (
    TSNE(n_components=2)
    .fit_transform(center_vector_matrix.numpy())
)
tsneDecode = (
    TSNE(n_components=2)
    .fit_transform(context_vector_matrix.numpy())
)


indexCount = 0 

plt.figure(figsize=(25, 5))

print("[INFO] Plotting TSNE Embeddings...")
for (word, embedding) in tsneEmbed[:100]:
    plt.scatter(word, embedding)
    plt.annotate(index_to_vocab[indexCount], (word, embedding))
    indexCount += 1
plt.savefig(config.SKIPGRAM_TSNE)
