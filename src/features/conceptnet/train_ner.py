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
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from rasa_nlu.training_data import load_data
import os
import numpy as np

dirname = os.path.dirname(__file__)
nlu_data_dir = os.path.join(dirname, '../../../data/processed')


# training data
TRAIN_DATA = [
    # ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    # ("I like London and Berlin.", {
    #  "entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]


def rasa_data_to_spacy_data(rasa_data):
    for m in rasa_data.training_examples:
        if not ("entities" in m.data):
            m.data["entities"] = []
        ents = [(e["start"], e["end"], e["entity"])
                for e in m.data["entities"]]
        s_d = (m.text, {"entities": ents})
        TRAIN_DATA.append(s_d)


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    train_data = load_data(
        nlu_data_dir + "/onto5/onto-nlu-data.json")
    train_data, _ = train_data.train_test_split(train_frac=0.1)
    rasa_data_to_spacy_data(train_data)
    """Load the model, set up the pipeline and train the entity recognizer."""
    # if model is not None:
    #     nlp = spacy.load(model)  # load existing spaCy model
    #     print("Loaded model '%s'" % model)
    # else:
    #     nlp = spacy.blank("en")  # create blank Language class
    #     print("Created blank 'en' model")
    # load existing spaCy model
    nlp_en = spacy.load('en')
    nlp_spacy = spacy.load('./data/embeddings/numberbatch/numberbatch_spacy')
    nlp = nlp_spacy
    print("Model loaded")
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        tag = nlp_en.get_pipe("tagger")
        pars = nlp_en.get_pipe("parser")
        ner = nlp.create_pipe("ner")
###

        nlp.add_pipe(tag, last=True)
        nlp.add_pipe(pars, last=True)

###
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    print(nlp.pipeline)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    print("Training...")
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            print(itn)
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
            print("Losses", losses)

    # test the trained model
    # for text, _ in TRAIN_DATA:
    #     doc = nlp(text)
    #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        # print("Loading from", output_dir)
        # nlp2 = spacy.load(output_dir)
        # for text, _ in TRAIN_DATA:
        #     doc = nlp2(text)
        #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]
