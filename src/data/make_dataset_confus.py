from __future__ import division
import sys
import json
import pprint
import numpy as np
from pathlib import Path
import os
import copy
import plac

dirname = os.path.dirname(__file__)
data_processed_dir = os.path.join(dirname, '../../data/processed')
stats_per_sf_dir = os.path.join(
    dirname, '../../data/external/stats_per_surface_form')


def get_stats_per_sf(name):
    with open(name) as data_file:
        data = json.load(data_file)

    total = 0
    number_of_resources = {}
    link_combo_frequency = {}
    confusing_entities = []
    for item in data:
        total = +1
        if data[item] > 1:
            confusing_entities.append(item)
        if data[item] in number_of_resources:
            number_of_resources[
                data[item]] = number_of_resources[data[item]] + 1
        else:
            number_of_resources[data[item]] = 1

    meanings = 0
    forms = 0
    confus_data = []
    for i in number_of_resources:
        print(i, number_of_resources[i])
        for x in range(0, int(number_of_resources[i])):
            confus_data.append(int(i))

    return confusing_entities, confus_data


@plac.annotations(
    dataset=("dataset", "positional", None, str),
    sf_data=("surface form data", "positional", None, str)
)
def main(dataset=None, sf_data=None):
    nlu_data = []
    with open(dataset) as data_file:
        nlu_data = json.load(data_file)["rasa_nlu_data"]["common_examples"]

    print("All examples: ", len(nlu_data))

    confusing_entities, confus_data = get_stats_per_sf(sf_data)
    # confusing_entities = confusing_entities + get_stats_per_sf(Path(stats_per_sf_dir) /
    #                                                            "neel2015.json")

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

    print("confusability mean: ", "%.2f" % np.mean(confus_data))
    print("confusability stdev: ", "%.2f" % np.std(confus_data))

    nlu_data = {"rasa_nlu_data": {
        "common_examples": nlu_data,
        "regex_features": [],
        "lookup_tables": [],
        "entity_synonyms": []
    }}

    name = dataset.split("/")[-1]

    with open(Path(data_processed_dir)/"confusing"/name, "w") as outfile:
        json.dump(nlu_data, outfile, separators=(',', ':'), indent=4)


if __name__ == "__main__":
    plac.call(main)
