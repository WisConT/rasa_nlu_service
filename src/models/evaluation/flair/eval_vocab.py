import sys
import os
import json
from flair.models import SequenceTagger

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../../'))  # NOQA: E402

from data.make_dataset_wnut import get_dataset as get_wnut
from data.make_dataset_conll import get_dataset as get_conll
from data.onto.make_dataset import get_dataset as get_onto
from baseline_model import get_statistics

def get_vocab_size(docs):

    vocab = []

    for doc in docs:
        for sentence in doc:
            for word in sentence['words']:
                if word not in vocab:
                    vocab.append(word)
    
    return len(vocab)

def evaluate():
    wnut = get_wnut()
    conll = get_conll()

    test, train, dev = get_onto(cased=False)

    onto = test + train + dev

    onto_vocab = get_vocab_size(onto)
    conll_vocab = get_vocab_size(conll)
    wnut_vocab = get_vocab_size(wnut)

    print(onto_vocab)
    print(conll_vocab)
    print(wnut_vocab)

if __name__ == '__main__':
    evaluate()