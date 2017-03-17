# Natural Language Processing with SKLearn

## Vectorizing
We need a way to represent the corpus in a way that allows analysis.  Let's use
a tiny corpus of text as an example

```python
corpus = ['document zero is full of text, text, text',
         'this is text from document one', ]
```

A `CountVectorizer` counts the number of times a word appears in a document.
Each word is a 'feature' for our analysis.  Because the number of words (or
features!) in the complete set of documents might be quite large, the vector is
first fitted with the corpus, then transfomed into the sparse matrix.

```python
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()
vector.fit(corpus)
sparse_matrix = vector.transform(corpus)
features = vector.get_feature_names()

print(features)
```

Here, we can see that document zero contains 3 occurences of the 6th feature,
which happens to be the word 'text'.  You can convert the sparse matrix to a
normal python array, but in practice, the array will be so large that this will
not be practical.

```python
print('The 6th feature is the word "' +features[6] + '"')
print()
print(sparse_matrix)
print()
print(sparse_matrix.todense())
```

## A Larger Corpus
Scikit-Learn provides a number of convenient resources for natural language
processing, including corpora tagged with parts-of-speech identifiers, and
divided into training and test sets.

Let's load the text from a couple of old-school internet news groups, then see
if we can make predictions about them

```python
from sklearn.datasets import fetch_20newsgroups

categories = ['rec.autos', 'rec.motorcycles','sci.med', 'sci.space']
twenty_train = fetch_20newsgroups(subset='train', 
                                  categories=categories, 
                                  shuffle=True, random_state=42)

twenty_test = fetch_20newsgroups(subset='test', 
                                 categories=categories, 
                                 shuffle=True, random_state=42)
```

Instead of calling `fit()`, then `transform()`, you can call `fit_transform()`
to perform both operations in a slightly more efficient manner.

```python
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()
sparse_matrix = vector.fit_transform(twenty_train.data)
n = len(vector.get_feature_names())

vector.get_feature_names()[25000:25010]
```

Simply counting the number of times a word appears in a document as the
`CountVectorizer` does is a very simple approach.  If a word appears more often
in a document simply because that document is much larger than the others in the
corpus, we might be mislead into thinking more of our test sentences belong to
that document.  It also means that common, uninformative words such as 'if' or
'and' have a larger influence than they should.

We can compensate for these effects by using the `TfidfTransformer`.

```python
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
sparse_matrix = transformer.fit_transform(sparse_matrix)
```

## Training a Classifier
Let's use a multinomial naive Bayes classifier.  It's good for classifying
discrete features, such as word counts in a body of text.

```python
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(sparse_matrix, twenty_train.target)
```

## A Quick Test
As a test, we'll take a new collection of sentences, vectorise them, then use
our trained classifier to try and guess which news group our new sentences might
have come from

```python
new_corpus = ['keep you hands on the wheel',
              'keep your hands on the handlebar',
              'goes like a rocket', 
              'things are rocky', 
              'adjust the rocker',]

new_sparse_matrix = vector.transform(new_corpus)
predictions = classifier.predict(new_sparse_matrix)

for doc, category in zip(new_corpus, predictions):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
```

----
## Puting It All Together

We can save ourselves a bit of work by assembling all of our components into a
pipeline, and fitting it to the training data

```python
from sklearn.pipeline import Pipeline

text_classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()),
])

text_classifier = text_classifier.fit(twenty_train.data, twenty_train.target)
```

Now, let's check our predictions

```python
import numpy as np

predictions = text_classifier.predict(twenty_test.data)
np.mean(predictions == twenty_test.target)
```

Looks like 96% of our guesses were correct.

We can try another classifier.  Can we do better with stochastic gradient
descent?

```python
from sklearn.linear_model import SGDClassifier
text_classifier = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                       alpha=1e-3, n_iter=5, random_state=42)), ])

text_classifier.fit(twenty_train.data, twenty_train.target)
predictions = text_classifier.predict(twenty_test.data)
np.mean(predictions == twenty_test.target)
```

A bit better, but nothing to write home about.

Let's look at the confusion matrix.  It will show the number of times we
correctly (and incorrectly!) classified each document in the test set.  An
example confusion matrix looks like this:

| n = 310   | predict feature 1 | predict feature 2 | predict feature 3 |
|-----------|------------------:|------------------:|------------------:|
|feature 1  |        100        |         3         |         2         |
|feature 2  |          5        |        95         |         3         |
|feature 3  |          5        |         2         |        95         |

```python
from sklearn import metrics
print(metrics.confusion_matrix(twenty_test.target, predictions))
print()
print(metrics.classification_report(twenty_test.target, 
                                    predictions, 
                                    target_names=twenty_test.target_names))

```