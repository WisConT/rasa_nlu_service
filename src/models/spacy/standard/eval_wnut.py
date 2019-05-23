import sys
import os
import json
from rasa_nlu.model import Interpreter

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../../'))  # NOQA: E402

from data.make_dataset_wnut import get_dataset
from baseline_model import get_statistics

def evaluate():
    documents = get_dataset()

    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, '../../../../../models/spacy/default/current')
    interpreter = Interpreter.load(model_path)

    spacy_labels = ['person', 'org', 'gpe', 'loc', 'product', 'work_of_art']
    # wnut_labels = ['corporation', 'creative-work', 'group', 'location', 'person', 'product']
    # mappings for spaCy type => wnut type
    mappings = {
        "gpe": ["location"],
        "loc": ["location"],
        "org": ["group", "corporation"],
        "work_of_art": ["creative-work"]
    }

    statistics = get_statistics(documents, interpreter, spacy_labels, mappings)

    print("statistics: ")
    print(json.dumps(statistics, indent=4))

    f = open("results/wnut.json", "w+")
    f.write(json.dumps(statistics, indent=4))
    f.close()

if __name__ == "__main__":
    evaluate()