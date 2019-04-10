from __future__ import division
from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu.evaluate import *
from rasa_nlu.training_data import load_data
from rasa_nlu import config
from collections import defaultdict
import tempfile
import json
import os
from operator import eq
from baseline_model import entities_equal

"""
Run crossvalidation with nlu-data.json given trained models using nlu-config.yml
(Will need to convert data documents to a nlu-data.json/.md format)
"""

dirname = os.path.dirname(__file__)
data_dir = os.path.join(dirname, './data')


def get_match_result(targets, predictions):
    wnut_mappings = {
        "gpe": ["location"],
        "loc": ["location"],
        "fac": ["location"],
        "org": ["group", "corporation"],
        "norp": ["group"],
        "work_of_art": ["creative-work"],
        "person": ["person"],
        "product": ["product"],
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

    match_cat = get_match_result(true_token_labels, predicted_labels)

    return {"target_labels": true_token_labels, "predicted_labels": predicted_labels, "match_cat": match_cat}


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
    aligned_predictions = []
    for ts, ps, tks in zip(targets, predictions, tokens):
        aligned_predictions.append(align_predictions(ts, ps, tks))
    return aligned_predictions


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

    if not extractors:
        return entity_results

    entity_targets = get_entity_targets(test_data)
    aligned_predictions = align_all_predictions(
        entity_targets, entity_predictions, tokens)
    print(align_predictions)
    err = 1 - sum([aligned_predictions[i]["match_cat"]
                   for i in range(0, len(aligned_predictions))])/len(aligned_predictions)

    for i, td in enumerate(test_data.training_examples):
        aligned_predictions[i]["text"] = td.text

    return {"json": json.dumps(aligned_predictions, separators=(',', ':'), indent=4), "err": err}


def crossvalidation(data_path, config_path, folds=10, verbose=False):
    """
    Perform cross validation given Rasa NLU data and pipeline configuration

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
    err = 0
    for idx, (train, test) in enumerate(generate_folds(folds, data)):
        interpreter = trainer.train(train)
        res = evaluate_test_data(interpreter, test)
        err = err + res["err"]
        if not verbose:
            continue
        print("fold: " + str(idx))
        print("result: " + res["json"])
        print("error: " + str(res["err"]))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Average error: " + str(err/folds))


def evaluate(data_path, config_path, sample_string):
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


if __name__ == "__main__":
    crossvalidation(data_dir + "/wnut-nlu-data.json",
                    data_dir + "/nlu-config.yml", folds=2, verbose=True)
    # evaluate(data_dir + "/jason-nlu-data.json",
    #          data_dir + "/nlu-config.yml", "This is America")
