# Bags of Words: Bigrams, Trigrams, N-grams

## Basic counting

When doing data science with text, the first challenge is usually to find a way to convert that text into meaningful numbers, to which standard tools (statistical methods, plots, machine learning algorithms) can then be applied. Treating the text as simply a "bag of words" is one way to do this conversion. The basic "bag of words" is nothing more than every word and an associated count of how many times it appears within a chunk of text.

Consider the two sentences:

```
[
    "The quick brown fox jumped over the lazy dog.",
    "Then the dog jumped over the fox."
]
```

Our bag of words would consist of:

```
{
    "the":    4,
    "quick":  1,
    "brown":  1,
    "fox":    2,
    "jumped": 2,
    "over":   2,
    "lazy":   1,
    "dog":    2,
    "then":   1
}
```

## The limits of looking at single words

In the above, we're dividing the text into single word units, or n-grams of `n=1`. This can be useful for many types of analysis, but it does lose important context. Consider the following two sentences:

```
[
    "The quick brown fox jumped over the lazy dog.",
    "The quick brown fox hasn't jumped over the lazy dog."
]
```

"Jumped" occurs the same number of times in each sentence, but the addition of "hasn't" in the 2nd sentence changes our understanding of the event. Sometimes we can't even get close to the real meaning looking at single words. For example:

```
[
    "The quick brown fox jumped over the lazy dog.",
    "The dog ran down Fox Street."
]
```

In this sentence, we're not even dealing with a fox and a dog. Here "fox" refers to a place simply named after an animal. However, if we were to change our n-gram length to 2 (otherwise known as a "bigram"), we'd be able to differentiate between the two:

```
[
    "the quick":   1,
    "quick brown": 1,
    "brown fox":   1,
    "fox jumped":  1,
    "jumped over": 1,
    "over the":    1,
    "the lazy":    1,
    "lazy dog":    1,
    "the dog":     1,
    "dog ran":     1,
    "ran down":    1,
    "down fox":    1,
    "fox street":  1
]
```

More meaning could be encoded into every unit by splitting the text into trigrams (`n=3`) or even larger n-grams. But even in the above bigram example, we start to see a tradeoff. Even though we do have the dog appearing in both events, there's not a single bigram token that captures that. Each token appears exactly once, giving us no indication that any one of them are more important than the others. The larger the n-gram units, the more meaningful each one will be, but the less likely they'll be to occur multiple times.

# Term-Document Matrix (TDM) / Document-Term Matrix (DTM)

If we wanted to represent the sentences separately, we could do so
