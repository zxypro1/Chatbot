import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    :param tokenized_sentence: Array
    :param all_words: Array of all the words
    :return: bag of words(the embedding of the sentence)
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
