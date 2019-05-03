from __future__ import division
from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu.evaluate import *
from rasa_nlu.training_data import load_data, TrainingData
from rasa_nlu import config
from collections import defaultdict
import tempfile
import json
import os
from operator import eq
import argparse
import plac
from pathlib import Path

"""
Run crossvalidation with nlu-data.json given trained models using nlu-config.yml
(Will need to convert data documents to a nlu-data.json/.md format)
"""

dirname = os.path.dirname(__file__)
nlu_data_dir = os.path.join(dirname, '../../data/processed')
nlu_resources_dir = os.path.join(dirname, './resources')

# MANY TO ONE MAPPING FOR WNUT
# spacy -> wnut
wnut_labels = ["location", "group",
               "creative-work", "person", "product", "o"]
wnut_mappings = {
    "gpe": "location",
    "loc": "location",
    "fac": "location",
    "org": "corporation",
    "norp": "group",
    "work_of_art": "creative-work",
    "person": "person",
    "product": "product",
    "event": "o",
    "law": "o",
    "language": "o",
    "date": "o",
    "time": "o",
    "percent": "o",
    "money": "o",
    "quantity": "o",
    "ordinal": "o",
    "cardinal": "o",
    "o": "o"
}

conll_mappings = {
    "gpe": ["loc"],
    "loc": ["loc"],
    "fac": ["loc"],
    "org": ["org"],
    "norp": ["o"],
    "work_of_art": ["o"],
    "person": ["per"],
    "product": ["o"],
    "event": ["o"],
    "law": ["o"],
    "language": ["o"],
    "date": ["o"],
    "time": ["o"],
    "percent": ["o"],
    "money": ["o"],
    "quantity": ["o"],
    "ordinal": ["o"],
    "cardinal": ["o"],
    "o": ["o"]
}


def get_match_result(targets, predictions):
    return all([t.lower() in wnut_mappings[p.lower()] for t, p in zip(targets, predictions)])

# def get_match_result(targets, predictions):
#     """
#     Metric found here:
#     http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

#     Needs improvement
#     """
#     if len(targets) < len(predictions):
#         return 2
#     elif len(targets) > len(predictions):
#         return 3
#     ent_match = True
#     loc_match = True
#     for t, p in zip(targets, predictions):
#         ent_match = t["entity"] == p["entity"] and ent_match
#         loc_match = t["start"] == p["start"] and t["end"] == p["end"] and loc_match
#     if ent_match and loc_match:
#         return 1
#     elif ent_match:
#         return 5
#     elif loc_match:
#         return 4
#     else:
#         return 6


def align_predictions(targets, predictions, tokens):
    """
    Aligns entity predictions to the message tokens. Determines for every token the true label based on the
    prediction targets and the label assigned (Repurposed from Rasa)

    Parameters:
        targets: list of target entities
        predictions: list of predicted entities
        tokens: original message tokens

    Returns:
        Dictionary containing the true token labels and token labels
    """
    true_token_labels = []
    predicted_labels = []

    for t in tokens:
        true_token_labels.append(determine_token_labels(t, targets, None))
        extracted = determine_token_labels(t, predictions, None)
        predicted_labels.append(extracted)

    return {"target_labels": true_token_labels, "predicted_labels": predicted_labels}


def align_all_predictions(targets, predictions, tokens):
    """
    Aligns entity predictions to the message tokens for the whole dataset using align_predictions (Repurposed from Rasa)

    Parameters:
        targets: list of lists of target entities
        predictions: list of lists of predicted entities
        tokens: list of original message tokens

    Returns:
        List of dictionaries containing the true token labels and token labels
    """
    prediction_per_document = []
    concat_targets = []
    concat_predictions = []

    for ts, ps, tks in zip(targets, predictions, tokens):
        aligned_json = align_predictions(ts, ps, tks)
        prediction_per_document.append(aligned_json)
        concat_targets = concat_targets + aligned_json["target_labels"]
        concat_predictions = concat_predictions + \
            aligned_json["predicted_labels"]

    return prediction_per_document, concat_targets, concat_predictions


def get_statistics(json_per_document, concat_targets, concat_predictions):
    from sklearn.metrics import precision_recall_fscore_support as pr
    prec, rec, f1, _ = pr([wnut_mappings[p.lower()] for p in concat_predictions],
                          ["group" if t.lower() is "corporation" else t.lower() for t in concat_targets], labels=wnut_labels)
    # prec, rec, f1, _ = pr([p.lower() for p in concat_predictions],
    #                       [wnut_mappings[t.lower()] for t in concat_targets], labels=wnut_labels)

    return {"precision": prec, "recall": rec, "f1": f1}


def evaluate_test_data(interpreter, test_data):
    """
    Evaluate test data given a trained interpreter

    Parameters:
        interpreter: trained interpreter
        test_data: data to be evaluated

    Returns:
        JSON object {"json", "err"}
    """
    entity_results = defaultdict(lambda: defaultdict(list))
    extractors = get_entity_extractors(interpreter)
    entity_predictions, tokens = get_entity_predictions(interpreter, test_data)
    print("got entity predictions")
    if not extractors:
        return entity_results

    entity_targets = get_entity_targets(test_data)
    print("got entity targets")

    per_document, all_t, all_p = align_all_predictions(
        entity_targets, entity_predictions, tokens)

    statistics = get_statistics(per_document, all_t, all_p)
    # err = 1 - sum([aligned_predictions[i]["match_cat"]
    #    for i in range(0, len(aligned_predictions))])/len(aligned_predictions)

    for i, td in enumerate(test_data.training_examples):
        per_document[i]["text"] = td.text

    return {"json": json.dumps(per_document, separators=(',', ':'), indent=4), "stats": statistics}


def crossvalidation(data_path, config_path, folds=10, verbose=False):
    """
    Perform cross validation given Rasa NLU data and pipeline configuration,
    printing f1 scores per entity

    Parameters:
        data_path: relative path of Rasa NLU data(.json)
        config_path: relative path of Rasa NLU pipeline config(.yml)
        folds: number of folds for cross validation
        verbose: if True, print out test data evaluation for each fold

    Returns:
        None
    """
    tmp_dir = tempfile.mkdtemp()
    trainer = Trainer(config.load(config_path))
    data = load_data(data_path)
    # err = 0
    for idx, (train, test) in enumerate(generate_folds(folds, data)):
        print("Training model with training data")
        interpreter = trainer.train(train)
        print("Evaluating test data with trained model")
        res = evaluate_test_data(interpreter, test)
        # err = err + res["err"]
        if not verbose:
            continue
        print("fold: " + str(idx))
        # print("result: " + res["json"])
        print("statistics: " + str(res["stats"]))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    # print("Average error: " + str(err/folds))


def evaluate(train_data_path, config_path, test_data_path):
    """
    Evaluate test data given model derived from Rasa NLU training data and pipeline configuration,
    printing f1 scores per entity

    Parameters:
        train_data_path: relative path of Rasa NLU training data(.json)
        config_path: relative path of Rasa NLU pipeline config(.yml)
        test_data_path: relative path of Rasa NLU test data(.json)

    Returns:
        None
    """
    trainer = Trainer(config.load(config_path))
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    print("Training model with training data")
    interpreter = trainer.train(TrainingData())
    print("Evaluating test data with trained model")
    res = evaluate_test_data(interpreter, test_data)
    print("statistics: " + str(res["stats"]))


def evaluate_string(data_path, config_path, sample_string):
    """
    Evaluate sample_string given model derived from Rasa NLU data and pipeline configuration

    Parameters:
        data_path: relative path of Rasa NLU data(.json)
        config_path: relative path of Rasa NLU pipeline config(.yml)
        sample_string: string to be evaluated

    Returns:
        None
    """
    trainer = Trainer(config.load(config_path))
    data = load_data(data_path)
    interpreter = trainer.train(data)
    raw = interpreter.parse(sample_string)
    print(json.dumps(raw, separators=(',', ':'), indent=4))


@plac.annotations(
    train_data_path=("training data", "positional", None, str),
    setting=("setting", "positional", None, str),
    config=("config", "positional", None, str),
    test_data_path=("test data", "positional", None, str),
    test_string=("string", "option", "string", None, str)
)
def main(train_data_path=None,  setting=None, config=None, test_data_path=None, test_string=None):
    if setting == "crossvalidation":
        crossvalidation(nlu_data_dir + train_data_path,
                        nlu_resources_dir + config, folds=2, verbose=True)
    elif setting == "evaluate":
        evaluate(nlu_data_dir + train_data_path,
                 nlu_resources_dir + config,
                 nlu_data_dir + test_data_path)
    elif setting == "evaluate_string":
        evaluate_string(nlu_data_dir + train_data_path,
                        nlu_resources_dir + config, test_string)


if __name__ == '__main__':
    plac.call(main)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="evaluation utilities")
#     parser.add_argument('--crossvalidation',
#                         help="use crossvalidation on data, printing statistics", action="store_true")
#     parser.add_argument('--evaluate',
#                         help="train model with training data and evaluate on test data, printing statistics", action="store_true")
#     parser.add_argument('--string',
#                         help="train model with training data and evaluate on string", action="store_true")
#     args = parser.parse_args()

#     if args.crossvalidation and args.evaluate:
#         print("Cannot use options --crossvalidation and --evaluate at the same time")
#     elif args.crossvalidation:
#         crossvalidation(nlu_data_dir + "/wnut_2017/wnut-train-nlu-data.json",
#                         nlu_resources_dir + "/nlu-config.yml", folds=2, verbose=True)
#     elif args.evaluate:
#         evaluate(nlu_data_dir + "/wnut_2017/wnut-train-nlu-data.json",
#                  nlu_resources_dir + "/nlu-config.yml",
#                  nlu_data_dir + "/wnut_2017/wnut-test-nlu-data.json")
#     elif args.string:
#         evaluate_string(nlu_data_dir + "/wnut_2017/wnut-train-nlu-data.json",
#                         nlu_resources_dir + "/similarity-config.yml", "I love the Arctic Monkeys, they are such a good band")
