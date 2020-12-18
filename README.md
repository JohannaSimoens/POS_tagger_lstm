# POS_tagger_lstm

POS tagger trained on Sequoia corpus with lstm (pytorch)

Given a Sequence of words (sentence), predict a sequence of part-of-speech tags.

- Model 1: LSTM using one-hot vectors to encode words
- Model 2: LSTM using pretrained word emceddings vectors to encode words

The baseline of this NLP task (Part-of-Speech tagging) is the Most Frequent Sense. 

# Requirements

- Python 3.8.5
- Pytorch

You need to download French word embeddings "vecs100-linear-frwiki" trained by M. Coavoux, via word2vec (skip-gram model) on the wikipedia dump (650 millions of words) frwiki-20140804-corpus.xml.bz2 (downloaded there http://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/):

http://www.linguist.univ-paris-diderot.fr/~mcandito/vecs100-linear-frwiki.bz2

And put this file in repository ./data_WSD_VS

NB: Other word embeddings are possible: the dimension of the word embeddings should be 100, and the file containing these word embeddings must be a text file with a word embedding per line, the token (word) and the float values (vector values, word embeddings) must be separated by spaces (first the token, then the float values).

# Corpus

https://deep-sequoia.inria.fr/

Corentin Ribeyre, Marie Candito, et Djamé Seddah. 2014. Semi-Automatic Deep Syn- tactic
Annotations of the French Treebank. Proceedings of the 13th International Workshop on Treebanks and
Linguistic Theories. Tübingen Universität, Tübingen, Germany
