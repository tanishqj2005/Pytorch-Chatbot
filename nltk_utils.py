from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    ans = np.zeros(len(all_words), dtype=np.float32)
    for id, word in enumerate(all_words):
        if word in tokenized_sentence:
            ans[id] = 1.0

    return ans

# Example 1:
# a = "How long does shipping take?"
# print(a)
# a = tokenize(a)
# print(a)


# Example 2:
# words = ["organize","Organizes","organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)
