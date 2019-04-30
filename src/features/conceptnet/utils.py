import pandas as pd
import numpy as np
import os
import sys
import json

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

import data.make_embeddings_csv_numberbatch as nb
from sklearn.metrics.pairwise import cosine_similarity

df = nb.get_embeddings_dataframe()
classes = ["person", "geopolitical_entity"]
classes_emb = nb.get_embeddings(df, classes)


def add_features(token_emb, classes_emb):
    if len(token_emb) == 0:
        return '0'
    per = cosine_similarity(
        classes_emb['/c/en/' + classes[0]].reshape(1, -1), token_emb.reshape(1, -1))[0][0]
    loc = cosine_similarity(
        classes_emb['/c/en/' + classes[1]].reshape(1, -1), token_emb.reshape(1, -1))[0][0]

    # clf = '0'
    # # hyperparams
    # if loc > 0.1 and loc - per > 0.1:
    #     clf = 'loc'
    # elif per > 0.1 and per - loc > 0.1:
    #     clf = 'per'

    return np.concatenate((token_emb, np.array([per, loc])))


def add_features_to_embeddings(df, tokens_emb, classes_emb):
    return {'/c/en/' + t: add_features(tokens_emb['/c/en/' + t], classes_emb) for t in tokens_emb.keys}


if __name__ == "__main__":
    # FOR NUMBERBATCH DISAMBIGUATION
    print("found dataframe")
    # sent = "Gotta dress up for london fashion week and party in style"
    # words = sent.split(" ")
    words = ["man"]
    tokens_emb = nb.get_embeddings(df, words)

    print(len(add_features_to_embeddings(
        df, tokens_emb, classes)['/c/en/' + "man"]))
