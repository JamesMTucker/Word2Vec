import os

# Number of dimensions
EMBEDDING_SIZE = 10

# Window size
WINDOW_SIZE = 5

ITERATIONS = 2000

# OUTPUT
OUTPUT_PATH = "outputs"

SKIPGRAM_LOSS = os.path.join(OUTPUT_PATH, 'loss_skipgram')
SKIPGRAM_TSNE = os.path.join(OUTPUT_PATH, 'tsne_skipgram')

CBOW_LOSS = os.path.join(OUTPUT_PATH, 'loss_cbow')
CBOW_TSNE = os.path.join(OUTPUT_PATH, 'tsne_cbow')

