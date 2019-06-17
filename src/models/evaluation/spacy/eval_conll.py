import sys
import os
import json
import spacy

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../../'))  # NOQA: E402

from data.make_dataset_conll import get_dataset
from baseline_model import get_statistics

def evaluate():
    documents = get_dataset()

    # dirname = os.path.dirname(__file__)
    # model_path = os.path.join(dirname, '../../../../models/spacy/onto_uncased/model-best')
    model_path = os.path.join(dirname, '../../../../models/spacy/onto_uncased/model-best')
    model = spacy.load(model_path)

    spacy_labels = ['person', 'org', 'gpe', 'loc']
    # conll_labels = ['per', 'org', 'loc']
    # mappings for spaCy type => conll type
    mappings = {
        "gpe": ["loc"],
        "person": ["per"]
    }

    statistics = get_statistics(documents, model, mappings, spacy_labels)

    print("statistics: ")
    print(json.dumps(statistics, indent=4))

    # f = open("results/conll_flair_fast.json", "w+")
    # f.write(json.dumps(statistics, indent=4))
    # f.close()

if __name__ == "__main__":
    evaluate()