# coding: UTF-8

import numpy as np


def read_embeddings(filename, verbose=0):
    """
    read embeddings file, returns an emdedding index to get embed float vector (=word embed) from a string (=word)
    """
    embedding_index = {}
    embedding_file = open(filename, 'r', encoding="utf-8")
    # header = list(map(int, embedding_file.readline().strip().split(' ')))
    for line in embedding_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    embedding_file.close()
    return embedding_index


def create_word_embeddings_matrix_from_corpus(word_lists, embedding_index):
    """
    Build vocabulary and weight matrix of word embeddings
    """
    weight_matrix_word_embeddings = []  # embedding matrix (list of lists) of dimension (vocab length, embedding dim)
    word_to_idx = {}  # dictionary to get word index (integer) from word (string)
    for word_list in word_lists:
        for word in word_list:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                if word in embedding_index:
                    weight_matrix_word_embeddings.append(embedding_index[word])
                else:  # if the word doesn't have a pre-trained word embed, just create zeros vector
                    weight_matrix_word_embeddings.append([0 for i in range(0, 100)])
    return weight_matrix_word_embeddings, word_to_idx

# test read_embeddings:
# embedding_index = read_embeddings("../data_WSD_VS/vecs100-linear-frwiki")
# print(len(embedding_index["le"]), embedding_index["le"])