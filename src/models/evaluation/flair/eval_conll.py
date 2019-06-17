import sys
import os
import json
from flair.models import SequenceTagger

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../../'))  # NOQA: E402

from data.make_dataset_conll import get_dataset
from baseline_model import get_statistics

def evaluate():
    documents = get_dataset()

    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, '../../../../models/flair/onto_uncased/crawl_flair_fast_embeddings/best-model.pt')

    model = SequenceTagger.load_from_file(model_path)

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

    f = open("results/conll_flair_fast.json", "w+")
    f.write(json.dumps(statistics, indent=4))
    f.close()

if __name__ == "__main__":
    evaluate()