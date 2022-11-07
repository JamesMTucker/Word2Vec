from tensorflow.keras.preprocessing.text import Tokenizer

def tensorflow_wrap(df):
    tokenizerSentence = Tokenizer()

    tokenizerLabel = Tokenizer()

    tokenizerSentence.fit_on_texts(df['sentence'])

    tokenizerLabel.fit_on_texts(df['sentiment'])

    encodedData = tokenizerSentence.texts_to_matrix(
        texts=df['sentence'], mode='count')

    labels = df['sentiment']

    for i in range(len(labels)):
        labels[i] = tokenizerLabel.word_index[labels[i]] - 1

    return (encodedData[:, 1:], labels.astype("float32"))