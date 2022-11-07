from pyimagesearch import config
from pyimagesearch.model import build_shallow_net
from pyimagesearch.bow import calculate_bag_of_words
from pyimagesearch.data_processing import preprocess
from pyimagesearch.data_processing import prepare_tokenizer
from pyimagesearch.tensorflow_wrapper import tensorflow_wrap
import pandas as pd

df = pd.DataFrame.from_dict(config.dataDict)

preprocessedDf = preprocess(sentDf=df, stopWords=config.stopWrds)
(textDict, labelDict) = prepare_tokenizer(df)

freqList = []

for sentence in df["sentence"]:
    entryFreq = calculate_bag_of_words(text=textDict, sentence=sentence)
    freqList.append(entryFreq)

finalDf = pd.DataFrame()

for vector in freqList:
    vector = pd.DataFrame([vector])
    finalDf = pd.concat([finalDf, vector], ignore_index=True)

finalDf["label"] = df["sentiment"]

for i in range(len(finalDf["label"])):
    finalDf["label"][i] = labelDict[finalDf["label"][i]]

shallowModel = build_shallow_net()
print("[INFO] compiling model...")

shallowModel.fit(
    finalDf.iloc[:,0:10],
    finalDf.iloc[:,10].astype("float32"),
    epochs=config.epochs,
    batch_size=config.batch_size
)

trainX, trainY = tensorflow_wrap(df)

tensorflowModel = build_shallow_net()
print("[INFO] compiling model with tensorflow wrapped data ...")

tensorflowModel.fit(
    trainX,
    trainY,
    epochs=config.epochs,
    batch_size=config.batch_size
)
