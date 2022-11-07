# Bag of Words Model

## Introduction

Bag of words model represents a words meaning as a vector of weights, based on the frequency of the words in the text. Thus, a word is represented in N-dimensional space, where N is the number of words in the vocabulary.

## Pros and Cons of BOW

As the vocabulary size is increased, the dimensionality of the vector increases. This is a problem when the dimensionality is too large. We therefore have spares vectors with zero weights.

The biggest problem is that the model doesn't represent the meaning of the words apropos the grammar and semantics of the language.


## Code

* Implementation of vanilla BOW model and Tensorflow
  *  [Train script](train.py)


## Bibliography

```bibtex
@incollection{Chakraborty_2022_BoW,
  author = {Devjyoti Chakraborty},
  title = {Introduction to the Bag-of-Words {(BoW)} Model},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/oa2kt},
}
```



#word2vec #bagofwords #tensorflow
