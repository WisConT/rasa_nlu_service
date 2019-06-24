from rasa_nlu.training_data import load_data
from pathlib import Path
import os
from nltk.stem.snowball import SnowballStemmer

dirname = os.path.dirname(__file__)
processed_dir = os.path.join(
    dirname, '../../data/processed/')

stemmer = SnowballStemmer("english")

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

# print(len(onto5_data.intersection(conll_data)))
print("Onto:", len(onto5_data), len(onto5_data.intersection(conll_data)), len(onto5_data.intersection(
    wnut_data)), len(onto5_data.intersection(neel_data)), len(onto5_data.intersection(newsreader_data)))
print("Conll:", len(conll_data.intersection(onto5_data)), len(conll_data), len(conll_data.intersection(
    wnut_data)), len(conll_data.intersection(neel_data)), len(conll_data.intersection(newsreader_data)))
print("Wnut:", len(wnut_data.intersection(onto5_data)), len(wnut_data.intersection(conll_data)), len(
    wnut_data), len(wnut_data.intersection(neel_data)), len(wnut_data.intersection(newsreader_data)))
print("Neel:", len(neel_data.intersection(onto5_data)), len(neel_data.intersection(conll_data)), len(
    neel_data.intersection(wnut_data)), len(neel_data), len(neel_data.intersection(newsreader_data)))
print("News:", len(newsreader_data.intersection(onto5_data)), len(newsreader_data.intersection(conll_data)), len(
    newsreader_data.intersection(wnut_data)), len(newsreader_data.intersection(neel_data)), len(newsreader_data))
