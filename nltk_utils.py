import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass
