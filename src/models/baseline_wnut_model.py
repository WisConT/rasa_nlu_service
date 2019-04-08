import sys
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../')
sys.path.append(filename)

from baseline_model import get_statistics
from data.make_dataset_wnut import parse_file

documents = parse_file(os.path.join(dirname, '../../data/external/wnut_2017/test.txt'))

spacy_labels = ['person', 'org', 'gpe', 'loc', 'product', 'work_of_art']
# wnut_labels = ['corporation', 'creative-work', 'group', 'location', 'person', 'product']
# mappings for spaCy type => wnut type
mappings = {
    "gpe": ["location"],
    "loc": ["location"],
    "org": ["group", "corporation"],
    "work_of_art": ["creative-work"]
}

(recall, precision, f1) = get_statistics(documents, spacy_labels, mappings)

print("recall: " + str(recall))
print("precision: " + str(precision))
print("f1: " + str(f1))