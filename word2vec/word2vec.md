# Word to Vector

An algorithm approach to representing a word's meaning in the form of a vector.

> Word2Vec essentially means expressing each word in your text corpus in an N-dimensional space (embedding space). The word's weight in each dimension of that embedding space defines it for the model.

The last sentence is worded horrible. We can restate it more intelligently as, "The model is made based on the weight of each word."

The meaning of a word is made by the distribution of a word within it's linguistic usage.

## Continuous Bag-of-Words (CBOW)

Predict the center word.

Example sentence:
`I am reading the book`

Window size of three:

* `I`, `reading` for label `am`
* `am`, `the` for the label `reading`
* `reading`, `book`, for the label `the`

![[CBOW.png]]


## Skip-Gram

Predict the neighboring words.

![[skip_gram.png]]


# Implementation


#word2vec #linearAlgebra #vector #matrix #CBOW