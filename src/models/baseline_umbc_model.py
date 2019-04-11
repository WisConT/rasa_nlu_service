import sys
import os

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

from data.make_dataset_umbc import get_dataset
from baseline_model import get_statistics
import json

documents = get_dataset()

spacy_labels = ['person', 'org', 'gpe', 'loc']
# umbc_labels = ['per', 'org', 'loc']
# mappings for spaCy type => umbc type
mappings = {
    "gpe": ["loc"],
    "loc": ["loc"],
    "person": ["per"],
    "org": ["org"]
}

statistics = get_statistics(documents, spacy_labels, mappings)

print("statistics: ")
print(json.dumps(statistics, indent=4))
