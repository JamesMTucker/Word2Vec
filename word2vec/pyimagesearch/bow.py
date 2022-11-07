def calculate_bag_of_words(text, sentence):
    freqDict = dict.fromkeys(text, 0)

    for word in sentence:
        if word in freqDict:
            freqDict[word] += sentence.count(word)
    
    return freqDict