
# Predicting location of a missing word through ngrams

This script goes through using n-grams to predict the location of a missing word in a sentence. The problem is based on a competition previously held on [Kaggle](https://www.kaggle.com/c/billion-word-imputation), to impute a singular word into a sentence . That problem can be solved in two subtasks: predicitng the missing word's location, and then inserting the most probable word. Here we attempt to model the former.


```python
import string 
import nltk
import random
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
```

The data can be found (at the time of writing) on the Kaggle competition page for [download](https://www.kaggle.com/c/billion-word-imputation/download/train_v2.txt.zip). The training data are approximately 4.15Gb after unzipping. 


```python
with open("data/train/train_v2.txt", 'r') as f:
    #f.readline() # skip header
    corpus = f.readlines()
# check cleanliness if need be
len(corpus)
# 30301028
```

The data come to us as complete sentences, with no particular ordering or themes, but thankfuly (presumably) cleaned and ready to work with.
For our proof of concept we will do just the necessary preprocessing to work with our sentences - worrying about more thourghough work at a later date.

Create our training set with a randomly removed word from each sentence, noting the word and it's location in the sentence. For the rules of this competition, the first and last word of a sentence could not be removed, BUT the last 'word' was always ".", which we strip out anyways with the rest of punctuation. For proof of concept and computing time, we will downsample to only the first 5000 sentences. 


```python
def pullword(l):
    """ Removes a random item from a tokenized list"""
    temp = l
    index = temp.index(random.Random(0).choice(temp[1:]))
    y_train.append(temp[index])
    y_train_index.append(index)

    temp.pop(index)
    X_train.append(temp)
```


```python
exclude = set(string.punctuation)
train = [nltk.word_tokenize("".join(ch for ch in line.strip("\n") if ch not in exclude)) for line in corpus[0:5000] ]   # sampledown train for proof of concept
X_train, y_train, y_train_index = [], [], []
for line in train[0:5000]: 
    if len(line) <= 2:
        train.pop(train.index(line))
    else:
        pullword(line)
```

Good. We now have 4 key data structures to run through our model: train, X_train (with a removed word), y_train (the removed word), and y_train_index.

## The Model

Predicting where a word is missing from a sentence can be done in multiple ways, including Parts of Speech (described elsewhere in this repo), and n-gram probability which we do here. 

Given the number of occurences of all bigrams C(w1,w2) and the occurences of all trigrams C(w,1,wx,w2), we calculate the number of occurences, D(w1,w,w2), where the is one and only one word inbetween w1 and w3. We can then apply this as a probability in our word-removed sentences, scoring which bigram is the most likely to actually be a trigram of the form D(w1,w,w2). 


```python
#TODO format formulae
```

To do this we must first set up our bigrams/trigrams, saving the output to persistence should we later revisit the model. This will become important once running a fuller data set.


```python
cv_bigram = CountVectorizer(tokenizer=lambda doc: doc, 
                               analyzer='word', 
                               input=u'content', 
                               ngram_range=(2,2), 
                               min_df=0.0, 
                               lowercase=False).fit(train)
bigrams = cv_bigram.transform(train)
joblib.dump(bigrams, "persistence/bigrams.pkl")
```


```python
cv_trigram = CountVectorizer(tokenizer=lambda doc: doc, 
                               analyzer='word', 
                               input=u'content', 
                               ngram_range=(3,3), 
                               min_df=0.0, 
                               lowercase=False).fit(train)
trigrams = cv_trigram.transform(train)
joblib.dump(bigrams, "persistence/trigrams.pkl")
```

Occurence counts can be seen quite quickly, but the sparse array implemented by scipy is not particularly useful for inspection. 


```python
# sum the occurences of C(w1, w2) bigrams
Cbigram = bigrams.sum(axis=0)
# sum the occurnces of C(w1, w2, w3) trigrams
Ctrigram = trigrams.sum(axis=0)
```

create D(w1,w2) , the number of occurences of the trigram of the form w1, w, w3 to use as out probability of a bigram actually being a trigram


```python
bgs = []
for item in cv_bigram.vocabulary_.items():
    l = item[0].split()
    words = (l[0], l[1])
    bgs.append(words)
      
tgs = []
for item in cv_trigram.vocabulary_.items():
    l = item[0].split()
    words = (l[0], l[1], l[2])
    tgs.append(words)   
```

Note that this is the largest step of creating our model, and takes a long time (many hours), even on the small proof-of-concept subsample. It is highly recommended to pickle the output once run, so it does not need to be recreated. 


```python
# form a dictionary of bigrams that could be the outer trigram
E = dict() # for the bigram (this is what we'll use )
for tg in tgs:
    for bg in bgs:
        if (tg[0] == bg[0]) and (tg[2] == bg[1]):
            bgram = " ".join(s for s in bg)
            tgram = " ".join(s for s in tg)
            index = cv_trigram.vocabulary_.get(" ".join(s for s in tg))
            if bgram not in E.keys():
                E[bgram] = [index] 
            else:
                if index not in E[bgram]:
                    E[bgram].append(index)
                else:
                    print("index: " + str(index) + " is already in E[" +str(bgram) +"]") # something weird has occurred.
joblib.dump(E, "persistence/E.pkl")
```

And create a score matrix for every bigram occurence as a possible trigram


```python
D = {}
for item in cv_bigram.vocabulary_:
    D[item] = 0
# sum the counts for each bigram in E by looking up in cv_bigrams
for item in E:
    for index in E[item]:
        D[item] += trigrams[:, index].sum()
# drop zeros
D2 = { k:v for k, v in D.items() if v }
joblib.dum(D2, "persistence/D2.pkl")
```

Finding the location of the missing word is now able to be performed. The idea is, that in each sentence, all bigrams can be compared, and the one with the highest probability of being a trigram is the location of the missing word. Note that this works because in our test, we _know_ that a word is missing, but if we didn't we could set a threshold or other metric instead.


```python
def trigram_probality(bigram_as_list):
    """Returns the probability of a single bigram to be a trigram, given the trained model."""
    bg = " ".join(bigram_as_list)
    if cv_bigram.vocabulary_.get(bg):    
        bg_index = cv_bigram.vocabulary_.get(bg)
        prob = D2[bg] / (bigrams[:, bg_index].sum() + D2[bg])
        print(bg, prob) 
        return D2[bg] / (bigrams[:, bg_index].sum() + D2[bg])
    else:
        return 0.0

```


```python
def bigrams_formatter(l):
    """Helper function to ensure the bigrams are passed correctly to trigram_probability."""
    bg = []
    for i in range(len(l)-1):
        bg.append( str(l[i] + " " + str(l[i+1])))
    return bg
```


```python
def score_sentence(sentence_as_list):
    """Creates a list of probability scores for a given sentence, returning the index of the most likely position."""
    if type(sentence_as_list) == str:
        sentence_as_list = sentence_as_list.split()
    s = []
    for bigram in bigrams_formatter(sentence_as_list):
        s.append(trigram_probality(bigram))
    print("bigram-is-trigram probability: ", s)
    # in the case that all are zero, we will pick the first word, to let us know the model is failing
    #if s.index(max(s)) == 0:
    return s.index(max(s))
```

And that's it. we can now test it out a bit:


```python
gram = "the of"
sentence = "The dog chased after the "
score_sentence(gram.split())
score_sentence(sentence.split())
```

## Accuracy evaluation

To test our accuracy, we can now use X_train and y_train_index to see how well this model predicts the location of a missing word... sorta! Remember that we used (almost) the same sentences to create our bigrams and trigrams counts, so we're double-dipping into our data. But if we don't get a good result here, then we won't anywhere else either, and will have to re-approach the model as a whole.


```python
guess = []
for sentence in X_train:
    guess.append(score_sentence(sentence))
guess = pd.Series(guess)    
missing = pd.Series(y_train_index)

result = pd.concat([guess, missing], axis=1)
result['hit'] = np.where(result['guess'] == result['missing'], 1, 0)

accuracy = result['hit'].sum() / len(result['hit']) * 100
```

#TODO input results table

## Conclusion

There you have it. We went through tokenizing a large text dataset, created massive (and sparse) arrays of vectorized counts, and created a bigram-trigram comparison model to predict where in a sentence a word is missing. 

The next step, is to impute the word, which again, can be done in a multitude of ways. 


```python

```
