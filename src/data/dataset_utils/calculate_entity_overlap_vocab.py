import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import os
from rasa_nlu.training_data import load_data
from pathlib import Path

dirname = os.path.dirname(__file__)
processed_dir = os.path.join(
    dirname, '../../data/processed/')

stemmer = SnowballStemmer("english")

nb_df = [r[1] for r in np.array(pd.read_csv(
    "./data/embeddings/numberbatch/numberbatch.csv").values)]
nb_df = set([e if type(e) is float else stemmer.stem(e) for e in nb_df])
nb_len = len(nb_df)
print("Numberbatch:", nb_len)
gl_df = [r[1] for r in np.array(pd.read_csv(
    "./data/embeddings/glove/glove.csv").values)]
gl_df = set([e if type(e) is float else stemmer.stem(e) for e in gl_df])
gl_len = len(gl_df)
print("GloVe:", gl_len)
print("Overlap:", len(nb_df.intersection(gl_df)))

onto5_data = load_data(str(Path(processed_dir) /
                           "onto5/onto-nlu-data.json")).sorted_entities()
onto5_data = set([stemmer.stem(e["value"].lower()) for e in onto5_data])

conll_data = load_data(str(Path(processed_dir) /
                           "conll_2003/conll-nlu-data.json")).sorted_entities()
conll_data = set([stemmer.stem(e["value"].lower()) for e in conll_data])

wnut_data = load_data(str(Path(processed_dir) /
                          "wnut_2017/wnut-nlu-data.json")).sorted_entities()
wnut_data = set([stemmer.stem(e["value"].lower()) for e in wnut_data])

neel_data = load_data(str(Path(processed_dir) /
                          "microposts2015/neel-nlu-data.json")).sorted_entities()
neel_data = set([stemmer.stem(e["value"].lower()) for e in neel_data])

newsreader_data = load_data(
    str(Path(processed_dir)/"newsreader/newsreader-nlu-data.json")).sorted_entities()
newsreader_data = set([stemmer.stem(e["value"].lower())
                       for e in newsreader_data])

print("Numberbatch:", len(nb_df.intersection(onto5_data)), len(nb_df.intersection(conll_data)), len(nb_df.intersection(
    wnut_data)), len(nb_df.intersection(neel_data)), len(nb_df.intersection(newsreader_data)))
print("GloVe:", len(gl_df.intersection(onto5_data)), len(gl_df.intersection(conll_data)), len(gl_df.intersection(
    wnut_data)), len(gl_df.intersection(neel_data)), len(gl_df.intersection(newsreader_data)))
