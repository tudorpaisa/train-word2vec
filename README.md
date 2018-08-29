# Script to train a Word2Vec model

As the title says, this is a simple script that trains a Word2Vec (W2V) model when given a __corpus__.

The W2V implementation is made using [gensim's Word2Vec](https://radimrehurek.com/gensim/), tune to highest known accuracy ([at least to my knowledge](https://arxiv.org/abs/1610.01520)). By this I mean that the result will be a **skip-gram model** with a **window size** = 15, **negative sampling** = 10, **vector size** = 300. In addition, this model will be trained on **trigrams**

## How to
You load up the script with:
```
python w2v.py
```

Once that is done, you will be asked to enter several things:

1. the full path to the corpus. Ex:

```
/home/USER/projects/mycorpus.txt 
```
2. How many threads you want to use. This one depends on your CPU; the more there are, the faster the training. (__NEEDS EXTRA DEPENDENCIES__; SEE NEXT SECTION)

3. The name of the file

When the corpus has been diced, munched, and sliced, you will see in the script's folder a file with the extension `.wv` and your desired name.

:exclamation: A quick note on the corpus: to properly work, you have to have to have it as a plain-text file (`.txt` for instance).

## Dependencies
```
python3 gensim nltk
```

To be able to use all your threads for the training, make sure you also have:
```
a_C_compiler cython
```

## Corpus tips
For best results, make sure your corpus is:
1. Over/At least 10 million results. If not, you're better off using LSA
2. Plain text

## WARNING
The skip-gram model takes a while to train (even with multi-threading). In addition, because this script allows for trigrams as well, the computation time will be doubly so. Therefore, make sure you have a CPU with many cores, or something else to do in the meantime.

## Credits
[Stanfordnlp's](https://github.com/stanfordnlp) stopword list from [CoreNLP](https://github.com/stanfordnlp/CoreNLP)

