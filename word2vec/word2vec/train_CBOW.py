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
with open("data_nietzsche.txt") as f:
    lines = f.readlines()
text_data = "".join(lines)

# print(text_data)

(vocab, tokenized_text_size, tokenized_text) = tokenize_data(text_data)

print(vocab)

# Map words to indices
vocab_to_index = {
    uniqueWord:index for (index, uniqueWord) in enumerate(vocab)
}
index_to_vocab = np.array(vocab)

text_as_int = np.array([vocab_to_index[word] for word in tokenized_text])

context_vector_matrix = tf.Variable(
    np.random.rand(tokenized_text_size, config.EMBEDDING_SIZE)
)

center_vector_matrix = tf.Variable(
    np.random.rand(tokenized_text_size, config.EMBEDDING_SIZE)
)

optimizer = tf.optimizers.Adam()
loss_list = []

print("[INFO] starting CBOW training ...")
for iter in tqdm(range(config.ITERATIONS)):
    loss_per_epoch = 0

    for start in range(tokenized_text_size - config.WINDOW_SIZE):
        indices = text_as_int[start:start + config.WINDOW_SIZE]

    with tf.GradientTape() as tape:
        combinded_context = 0

        for count, index in enumerate(indices):
            if count != config.WINDOW_SIZE // 2:
                combinded_context += context_vector_matrix[index, :]
        
        combinded_context /= (config.WINDOW_SIZE - 1)

        output = tf.matmul(center_vector_matrix, tf.expand_dims(combinded_context, 1))

        softout = tf.nn.softmax(output, axis=0)
        loss = softout[indices[config.WINDOW_SIZE // 2]]

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

print("[INFO] Plotting loss ...")
plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(config.CBOW_LOSS)

tsne_embed = (
    TSNE(n_components=2)
    .fit_transform(center_vector_matrix.numpy())
)
tsne_decode = (
    TSNE(n_components=2)
    .fit_transform(context_vector_matrix.numpy())
)

index_count = 0
plt.figure(figsize=(25, 5))
print("[INFO] Plotting TSNE embeddings ...")
for (word, embedding) in tsne_decode[:100]:
    plt.scatter(word, embedding)
    plt.annotate(index_to_vocab[index_count], (word, embedding))
    index_count += 1
plt.savefig(config.CBOW_TSNE)