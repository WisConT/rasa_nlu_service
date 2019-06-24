import warnings
from rasa_nlu.training_data import load_data
import os
import json
from pathlib import Path

dirname = os.path.dirname(__file__)
processed_dir = os.path.join(
    dirname, '../../../data/processed/')

warnings.filterwarnings("ignore")


def _get_dataset_count(path):
    for f in os.listdir(path):
        if ".DS_Store" in f or ".py" in f:
            continue
        data = load_data(str(Path(path)/f))
        print(f, len(data.training_examples))


def get_count():
    for nd in os.listdir(processed_dir):
        if ".DS_Store" in nd or "surface_form" in nd or "confusing" in nd:
            continue
        _get_dataset_count(Path(processed_dir)/nd)


def _get_entity_proportions(path):
    for f in os.listdir(path):
        if ".DS_Store" in f or ".py" in f:
            continue
        data = load_data(str(Path(path)/f))
        print(f, data.examples_per_entity)


def get_entity_proportions():
    for nd in os.listdir(processed_dir):
        if ".DS_Store" in nd or "surface_form" in nd or "confusing" in nd:
            continue
        _get_entity_proportions(Path(processed_dir)/nd)


if __name__ == "__main__":
    get_entity_proportions()
