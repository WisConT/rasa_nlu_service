import sys
import os
import json

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../../../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

from data.make_dataset_wnut import parse_file

documents = parse_file(os.path.join(
    dirname, '../../../../data/external/wnut_2017/test.txt'))

nlu_data = {"rasa_nlu_data": {
    "common_examples": [],
    "regex_features": [],
    "lookup_tables": [],
    "entity_synonyms": []
}}
#print(json.dumps(documents[0:2], separators=(',', ':'), indent=4))
common_examples = [{"intent": "greeting", "entities": doc["entities"], "text": doc["full_text"]}
                   for [doc] in documents]
nlu_data["rasa_nlu_data"]["common_examples"] = common_examples

# print(json.dumps(nlu_data, separators=(',', ':'), indent=4))

with open(dirname + '/wnut-nlu-data.json', 'w') as outfile:
    json.dump(nlu_data, outfile, separators=(',', ':'), indent=4)
