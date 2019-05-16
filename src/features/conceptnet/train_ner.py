#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function, division

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy._ml import Tok2Vec
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from rasa_nlu.training_data import load_data
import os
import numpy as np
import pandas as pd

dirname = os.path.dirname(__file__)
nlu_data_dir = os.path.join(dirname, '../../../data/processed')


TRAIN_DATA = []

VALID_DATA = []
# spaCy data example
# ("I like London and Berlin.", {
#  "entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),


def get_scores(nlp, examples):
    random.shuffle(examples)

    scores, loss = evaluate(nlp, examples)
    # report_scores(loss, scores)
    return scores, loss


def evaluate(nlp, dev_sents):
    scorer = Scorer()
    loss = 0
    for raw_text, annotations in dev_sents:
        doc = nlp.make_doc(raw_text)
        # ner(doc)
        gold = GoldParse(doc, entities=annotations['entities'])
        loss += gold.loss
        # nlp.tagger(doc)
        nlp.entity(doc)
        # nlp.parser(doc)
        scorer.score(doc, gold)
    return scorer.scores, np.reciprocal(scorer.scores['ents_f'])


# def report_scores(loss, scores):
#     """
#     prints precision recall and f_measure
#     :param scores:
#     :return:
#     """
#     precision = '%.2f' % scores['ents_p']
#     recall = '%.2f' % scores['ents_r']
#     f_measure = '%.2f' % scores['ents_f']
#     # print('%s %s %s %s' % (loss, precision, recall, f_measure))


def rasa_data_to_spacy_data(rasa_data, spacy_data):
    for m in rasa_data.training_examples:
        if not ("entities" in m.data):
            m.data["entities"] = []
        ents = [(e["start"], e["end"], e["entity"])
                for e in m.data["entities"]]
        s_d = (m.text, {"entities": ents})
        spacy_data.append(s_d)


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    train_loss = []
    valid_loss = []
    # pd.DataFrame(np.array([train_loss, valid_loss])).to_csv("./file.csv")
    data = load_data(
        nlu_data_dir + "/onto5/onto-train-nlu-data.json")
    train_data, test_data = data.train_test_split(train_frac=0.9)  # 0.9
    valid_data, _ = test_data.train_test_split(train_frac=1)  # 1
    rasa_data_to_spacy_data(train_data, TRAIN_DATA)
    rasa_data_to_spacy_data(valid_data, VALID_DATA)

    tok2vec_args = {'dep': False, 'sem_diff': False,
                    'width': 96, 'static': False}

    print("Loading model...")
    nlp_en = spacy.load('en')
    nlp = nlp_en
    if model == "numberbatch":
        nlp_nb = spacy.load('./models/language/numberbatch_spacy')
        nlp = nlp_nb
        print("Numberbatch language model loaded")
    elif model == "en_lg":
        nlp_lg = spacy.load('en_core_web_lg')
        nlp = nlp_lg
        print("en_core_web_lg model loaded")
    elif model == "en_md":
        nlp_md = spacy.load('en_core_web_md')
        nlp = nlp_md
        print("en_core_web_md model loaded")
    elif model == "en_sm":
        nlp_sm = spacy.load('en_core_web_sm')
        nlp = nlp_sm
        print("en_core_web_sm model loaded")
    else:
        print("Unsupported language model")
        exit()

    nlp.Defaults.create_vocab(nlp._meta.get(
        "vocab", {}), sem_diff=tok2vec_args["sem_diff"])

    tag = nlp_en.get_pipe("tagger")
    pars = nlp_en.get_pipe("parser")
    ner = nlp.create_pipe("ner")

    print("Add or replace pipes...")
    if "tagger" in nlp.pipe_names:
        nlp.replace_pipe("tagger", tag)
    else:
        nlp.add_pipe(tag, last=True)

    if "parser" in nlp.pipe_names:
        nlp.replace_pipe("parser", pars)
    else:
        nlp.add_pipe(pars, last=True)

    if "ner" in nlp.pipe_names:
        nlp.replace_pipe("ner", ner)
    else:
        nlp.add_pipe(ner, last=True)

    print("Pipeline:", nlp.pipeline)

    # add labels for transition system actions
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):
        nlp.begin_training(tok2vec_args=tok2vec_args)
        print('Training...')
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print(itn, "Training loss:", losses["ner"])
            train_loss.append(losses["ner"])

            if itn % 2 is 0:
                _, loss = get_scores(nlp, VALID_DATA)
                print("Validation loss: ", loss)
                valid_loss.append(loss)
            a = [] if len(valid_loss) < 4 else valid_loss[-4:]
            if len(a) > 0 and a[0] < a[1] and a[1] < a[2] and a[2] < a[3]:
                print("Overfitting")
                break

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    pd.DataFrame(np.array([train_loss, valid_loss])).to_csv("./file.csv")
    # plot thing


if __name__ == "__main__":
    plac.call(main)
