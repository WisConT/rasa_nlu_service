import sys
import os

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

from data.make_dataset_conll import parse_file
from baseline_model import get_statistics

documents = parse_file(os.path.join(
    dirname, '../../data/external/conll_2003/test.txt'))

spacy_labels = ['person', 'org', 'gpe', 'loc']
# conll_labels = ['per', 'org', 'loc']
# mappings for spaCy type => conll type
mappings = {
    "gpe": ["loc"],
    "person": ["per"]
}

(recall, precision, f1) = get_statistics(documents, spacy_labels, mappings)

print("recall: " + str(recall))
print("precision: " + str(precision))
print("f1: " + str(f1))
