import sys
import os
import json
import plac
import numpy as np

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402


@plac.annotations(
    setting=("setting", "positional", None, str),
    file_name=("txt", "positional", None, str)
)
def main(setting=None, file_name=None):
    nlu_data = {"rasa_nlu_data": {
        "common_examples": [],
        "regex_features": [],
        "lookup_tables": [],
        "entity_synonyms": []
    }}
    dataset_name = setting
    if setting == "wnut_2017":
        from data.make_dataset_wnut import parse_file
        txt_file_name = file_name
        documents = parse_file(os.path.join(
            dirname, '../../../data/interim/' + dataset_name + '/' + txt_file_name))
        prefix = "/wnut-"
        common_examples = [{"intent": "greeting", "entities": doc["entities"], "text": doc["full_text"]}
                           for [doc] in documents]
        nlu_data["rasa_nlu_data"]["common_examples"] = common_examples

        with open(dirname + '/../../../data/processed/' + dataset_name + prefix + os.path.splitext(txt_file_name)[0] + '-nlu-data.json', 'w') as outfile:
            json.dump(nlu_data, outfile, separators=(',', ':'), indent=4)
    elif setting == "onto5":
        from data.make_dataset_onto import get_dataset
        documents = get_dataset()
        prefix = "/onto"
        common_examples = [{"intent": "greeting", "entities": doc["entities"], "text": doc["full_text"]}
                           for doc in np.concatenate(documents)]
        nlu_data["rasa_nlu_data"]["common_examples"] = common_examples
        with open(dirname + '/../../../data/processed/' + dataset_name + prefix + '-nlu-data.json', 'w') as outfile:
            json.dump(nlu_data, outfile, separators=(',', ':'), indent=4)


if __name__ == '__main__':
    plac.call(main)
