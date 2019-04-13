import sys
import os
import json

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

from data.make_dataset_wnut import parse_file

dataset_name = sys.argv[1]
txt_file_name = sys.argv[2]

documents = parse_file(os.path.join(
    dirname, '../../../data/interim/' + dataset_name + '/' + txt_file_name))

nlu_data = {"rasa_nlu_data": {
    "common_examples": [],
    "regex_features": [],
    "lookup_tables": [],
    "entity_synonyms": []
}}

common_examples = [{"intent": "greeting", "entities": doc["entities"], "text": doc["full_text"]}
                   for [doc] in documents]
nlu_data["rasa_nlu_data"]["common_examples"] = common_examples

with open(dirname + '/../../../data/processed/' + dataset_name + '/wnut-' + os.path.splitext(txt_file_name)[0] + '-nlu-data.json', 'w') as outfile:
    json.dump(nlu_data, outfile, separators=(',', ':'), indent=4)
