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

"""
Run crossvalidation with nlu-data.json given trained models using nlu-config.yml
"""

dirname = os.path.dirname(__file__)
data_dir = os.path.join(dirname, './data')

def get_match_result(targets, predictions):
    """
    Metric found here:
    http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    """
    if len(targets) < len(predictions):
        return 2
    elif len(targets) > len(predictions):
        return 3

    ent_match = True
    loc_match = True
    for t, p in zip(targets, predictions):
        ent_match = t["entity"] == p["entity"] and ent_match
        loc_match = t["start"] == p["start"] and t["end"] == p["end"] and loc_match
    if ent_match and loc_match:
        return 1
    elif ent_match:
        return 5
    elif loc_match:
        return 4
    else:
        return 6
    
def align_predictions(targets, predictions, tokens):
    true_token_labels = []
    predicted_labels = []
    match_cat = get_match_result(targets,predictions)

    for t in tokens:
        true_token_labels.append(determine_token_labels(t, targets, None))
        extracted = determine_token_labels(t, predictions, None)
        predicted_labels.append(extracted)

    return {"target_labels": true_token_labels, "predicted_labels": predicted_labels, "match_cat": match_cat}

def align_all_predictions(targets, predictions, tokens):
    aligned_predictions = []
    for ts, ps, tks in zip(targets, predictions, tokens):
        aligned_predictions.append(align_predictions(ts, ps, tks))
    return aligned_predictions

def evaluate_test_data(interpreter, test_data):
    entity_results = defaultdict(lambda: defaultdict(list))
    extractors = get_entity_extractors(interpreter)
    entity_predictions, tokens = get_entity_predictions(interpreter, test_data)

    if not extractors:
        return entity_results

    entity_targets = get_entity_targets(test_data)
    aligned_predictions = align_all_predictions(entity_targets, entity_predictions, tokens)
    err = 1 - sum([aligned_predictions[i]["match_cat"] == 1 
                for i in range(0, len(aligned_predictions))])/len(aligned_predictions)

    for i, td in enumerate(test_data.training_examples):
       aligned_predictions[i]["text"] = td.text
                                
    return {"json": json.dumps(aligned_predictions,separators=(',', ':'),indent=4), "err": err}

def evaluate_model(data_path,config_path, valid_path):
    trainer = Trainer(config.load(config_path))
    data = load_data(data_path)
    valid = load_data(valid_path)
    interpreter = trainer.train(data)
    res = evaluate_test_data(interpreter,valid)
    return res["err"]


def crossvalidation(data_path, config_path, folds=10, verbose=False):
    tmp_dir = tempfile.mkdtemp()
    trainer = Trainer(config.load(config_path))
    data = load_data(data_path)
    err = 0
    for idx, (train, test) in enumerate(generate_folds(folds, data)):
        interpreter = trainer.train(train)
        res = evaluate_test_data(interpreter,test)
        err = err + res["err"]
        if not verbose: 
            continue
        print ("fold: " + str(idx))
        print ("result: " + res["json"])
        print ("error: " + str(res["err"]))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(err/folds)

def evaluate(data_path,config_path):
    trainer = Trainer(config.load(config_path))
    data = load_data(data_path)
    interpreter = trainer.train(data)
    raw = interpreter.parse(u"Australia is a country") # parses provided input and gives intent classification
    print (json.dumps(raw,separators=(',', ':'),indent=4))


crossvalidation(data_dir + "/jason-nlu-data.json", data_dir + "/nlu-config.yml", verbose=True)
#evaluate("./data/jason-nlu-data.json", "./data/nlu-config.yml")
