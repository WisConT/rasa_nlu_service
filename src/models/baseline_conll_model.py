import sys
import os
import json

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

from data.make_dataset_conll import get_dataset
from baseline_model import get_statistics

documents = get_dataset()

spacy_labels = ['person', 'org', 'gpe', 'loc']
# conll_labels = ['per', 'org', 'loc']
# mappings for spaCy type => conll type
mappings = {
    "gpe": ["loc"],
    "person": ["per"]
}

statistics = get_statistics(documents, spacy_labels, mappings)

print("statistics: ")
print(json.dumps(statistics, indent=4))
