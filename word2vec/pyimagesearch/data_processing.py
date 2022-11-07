import re


def preprocess(sentDf, stopWords, key='sentence'):
    """Loop over the sentences"""
    for num in range(len(sentDf[key])):
        """Loop over the words in the sentence"""
        sentence = sentDf[key][num]
        sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence.lower()).split()
        newWords = []
        for word in sentence:
            if word not in stopWords:
                newWords.append(word)
        sentDf[key][num] = newWords
    return sentDf


def prepare_tokenizer(df, sentKey="sentence", outputKey="sentiment"):
    wordCounter = 0
    labelCounter = 0

    textDict = dict()
    labelDict = dict()

    for entry in df[sentKey]:
        for word in entry:
            if word not in textDict.keys():
                textDict[word] = wordCounter
                wordCounter += 1
        for label in df[outputKey]:
            if label not in labelDict.keys():
                labelDict[label] = labelCounter
                labelCounter += 1
    return (textDict, labelDict)