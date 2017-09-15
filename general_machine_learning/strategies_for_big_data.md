## What is "Big Data"?

Big data can be a pretty subjective thing. When does it become "big"? When you gather enough of it that deriving insight from individual cases becomes less valuable than statistical insights that generalize over the entire dataset? Or when you can no longer fit that data into a spreadsheet? How about when you can no longer fit it on your laptop/desktop?

Or, perhaps the data itself isn't large in terms of file size. Perhaps it becomes "big" the moment you try to use it, due to various transformations and/or calculations that can take a long time to turn it into a useful state.

For the purposes of this section, we'll define "big data" as a point in working with data when the experimental feedback loop – having a hypothesis you'd like to test, getting the results, and deciding what to do next – becomes uncomfortably long.

This will obviously still be a bit subjective. Some problems, though they may take a long time, won't require a lot of iterations. It may not be worth the investment in setup time and infrastructure to treat these as "big data" problems. Some of the techniques/tools below may still make sense to try in these cases, but payoff vs investment may not be high enough for others.

## Vectorization

Vectorization is the practice of removing explicit `for` loops when operating over elements in vectors or matrices, preferring functions that can express a trasformation over the entire group of elements.

There's nothing magical about this. Somewhere in the guts of whatever API you use to do this, you're still iterating over the individual elements of the array. And, if the language is particularly good at optimizing your `for` loops, it may actually be less efficient, or just not worth the effort, to vectorize your code. Julia, for example, claims that `for` loops can often be as fast as "vectorized" code.

What you are really doing when you "vectorize" code is giving up control of *how* the operation is carried out over your entire array to whatever API you are using. That allows you to benefit from usually highly optimized code. The implementation may decide to run those operations in parallel before finally combining them into the output array. It may opt for a more complex loop in order to make fewer memory allocations. It may send the work to a GPU. Depending on certain numerical properties of your data, there may be calculations that can skip some of the usual steps.

## Sparse Matrices

Many problems that deal with discrete inputs and/or outputs will require one-hot encoding data. This can result in very large matrices, filled mostly with 0s. Instead of explicitly storing every 0, a sparse matrix will only store the indices of the non-zero values and assume the rest are 0. This can result in big savings in memory or disk storage. In some cases, it may even be possible to run optimized functions on sparse matrices.

## Parallelization

Any division of tasks that allows you to do operations independently of one another (even if you'll eventually have to combine the results) may allow much quicker computation. Vectorization can help with automatic parallelization, especially combined with GPU processing. Other tasks may be able to be explicitly parallelized.

## Cacheing / Data Centralization

Cacheing intermediate steps can save time if those steps take a long time to generate data and it doesn't change much. It is important to be able to update the cached data easily if the way it's processed does change, but depending on how infrequent that is, cacheing can cut out a lot of the wait time to test things further down the line.

Data centralization involves uploading source data and/or cached data to somewhere accessible by all members of the team. How the data is stored (whether on a traditional file system, object storage, or a database of some sort) doesn't matter as much as the fact that it is available to everyone. With this in place, only one person really has to spend the time creating some of the intermediate steps, and others can pull from the "cached" versions.

## Keras Techniques

### Batches and Generators

Keras' `fit` function can do many common tasks necessary in training on a data set for you *if* that dataset can fit in memory in the form that the `fit` function will consume it. One of the more common tasks involves dividing the training data into representative (usually by shuffling) mini-batches.

When either the dataset or the transformations required on the dataset to get it into a form you can pass to your training function are too big to be held in memory, Keras recommends doing your own batching of the data.

Although it works to simply loop over your batches, calling the `fit` function explicitly on every one, you end up generating a lot of extra output and it's harder to track the progress over batches and, ultimately, epochs during training.

Keras provides [`train_on_batch` and `test_on_batch` functions](https://keras.rstudio.com/articles/faq.html#batch-functions) to explicitly train on a single batch, skipping the extra overhead of the `fit` function. It also provides a [`fit_generator` function](https://keras.rstudio.com/articles/faq.html#external-data-generators). If you can craft your batch generation in a way that conforms to Keras' `generator` specification (read more about it [here](https://keras.io/models/model/)) and provide some extra information, such as `steps_per_epoch`, you can get most of the benefits of the `fit` function while being able to manage the batches yourself. Furthermore, Keras can do some processing of data for the next batches in the generator *parallel to* training the model on the current batch, provided there are extra CPU (or GPU) threads to support this.