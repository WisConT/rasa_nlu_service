from __future__ import division
import sys
import json
import pprint
import numpy as np
from pathlib import Path
import os
import copy

dirname = os.path.dirname(__file__)
data_processed_dir = os.path.join(dirname, '../../data/processed')
stats_per_sf_dir = os.path.join(dirname, '../../data/external/stats_per_surface_form')

def get_stats_per_sf(name):
    with open(name) as data_file:    
        data = json.load(data_file)

    total = 0 
    number_of_resources = {}
    link_combo_frequency = {}
    confusing_entities = []
    for item in data:
        total=+1 
        if len(data[item]) > 1:
            confusing_entities.append(item)
        if len(data[item]) in number_of_resources:
            number_of_resources[len(data[item])] = number_of_resources[len(data[item])] + 1 
        else:
            number_of_resources[len(data[item])] = 1 

    meanings = 0
    forms = 0
    confus_data = []
    for i in number_of_resources:
        print(i, number_of_resources[i])
        for x in range (0, int(number_of_resources[i])):
            confus_data.append(int(i))

    return confusing_entities, confus_data

def main():
    
    path_to_sf_data = Path(stats_per_sf_dir)/"AIDA-YAGO2_entities_and_links.json"
    nlu_data = []
    with open(Path(data_processed_dir)/"conll_2003"/"conll-nlu-data.json") as data_file:    
        conll_nlu_data = json.load(data_file)
        nlu_data = nlu_data + conll_nlu_data["rasa_nlu_data"]["common_examples"]
    with open(Path(data_processed_dir)/"onto5"/"onto-test-nlu-data.json") as data_file:    
        onto_nlu_data = json.load(data_file)
        nlu_data = nlu_data + onto_nlu_data["rasa_nlu_data"]["common_examples"]

    print("All examples: ", len(nlu_data))

    confusing_entities, _ = get_stats_per_sf(path_to_sf_data)
    confusing_entities = [ent.lower() for ent in confusing_entities]
    remove_indices = []
    for idx, ex in enumerate(nlu_data):
        if "entities" not in ex.keys():
            continue
        if all([ent["value"].lower() not in confusing_entities for ent in ex["entities"]]):
            remove_indices.append(idx)

    for idx in sorted(remove_indices, reverse=True):
        del nlu_data[idx]

    print("Confusing examples: ", len(nlu_data))

    nlu_data = {"rasa_nlu_data": {
        "common_examples": nlu_data,
        "regex_features": [],
        "lookup_tables": [],
        "entity_synonyms": []
    }}

    with open(Path(data_processed_dir)/"confusing"/"confusing-nlu-data.json", "w") as outfile:    
        json.dump(nlu_data, outfile, separators=(',', ':'), indent=4)
    

if __name__ == "__main__":
    main()