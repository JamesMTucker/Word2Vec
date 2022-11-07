dataDict = {
    "sentence":[
        "Avengers is a great movie.",
        "I love Avengers it is great.",
        "Avengers is a bad movie.",
        "I hate Avengers.",
        "I didnt like the Avengers movie.",
        "I think Avengers is a bad movie.",
        "I love the movie.",
        "I think it is great."
    ],
    "sentiment":[
        "good",
        "good",
        "bad",
        "bad",
        "bad",
        "bad",
        "good",
        "good"
    ]
}

stopWrds = ['is', 'a', 'i', 'it']

epochs = 30
batch_size = 10

denseUnits = 50
