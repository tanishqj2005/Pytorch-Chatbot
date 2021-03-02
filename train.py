import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))

X_train = []
Y_train = []

for (pattern,tag) in xy:
    bag = bag_of_words(pattern,all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    Y_train.append(label) #Cross entropy loss

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(X_train)
