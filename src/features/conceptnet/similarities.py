import typing
from typing import Any, Dict, List, Text
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message
import os
import sys
from rasa_nlu.components import Component
from rasa_nlu import utils
from rasa_nlu.model import Metadata
import spacy

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

import data.make_embeddings_csv_numberbatch as nb
import features.conceptnet.utils as cn

import nltk
from nltk.classify import NaiveBayesClassifier

SIMILARITY_MODEL_FILE_NAME = "similarity_classifier.pkl"

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyNBEntityExtractor(EntityExtractor):

    provides = ["entities"]

    requires = ["spacy_nlp"]

    defaults = {
        # by default all dimensions recognized by spacy are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None
    }

    def __init__(self, component_config: Text = None) -> None:
        self.nlp = spacy.load(
            # './models/pipeline/onto5/numberbatch/nb_onto5_large_vec300_proj96_model/')
            # './models/pipeline/en_sm/sm_onto5_large_sem_diff/')
            # './models/pipeline/conll_2003/numberbatch/nb_conll_large_sem_diff/61')
            './models/pipeline/conll_2003/numberbatch/nb_conll_large/75')
        # './models/pipeline/conll_2003/en_lg/en_lg_conll_pruned_sem_diff/')
        super(SpacyNBEntityExtractor, self).__init__(component_config)

    def process(self, message: Message, **kwargs: Any) -> None:
        # can't use the existing doc here (spacy_doc on the message)
        # because tokens are lower cased which is bad for NER
        spacy_nlp = self.nlp
        doc = spacy_nlp(message.text)
        all_extracted = self.add_extractor_name(self.extract_entities(doc))
        dimensions = self.component_config["dimensions"]
        extracted = self.filter_irrelevant_entities(
            all_extracted, dimensions
        )
        message.set(
            "entities", message.get("entities", []) + all_extracted, add_to_output=True
        )

    @staticmethod
    def extract_entities(doc: "Doc") -> List[Dict[Text, Any]]:
        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "confidence": None,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        return entities

    @staticmethod
    def filter_irrelevant_entities(extracted, requested_dimensions):
        """Only return dimensions the user configured"""

        if requested_dimensions:
            return [entity
                    for entity in extracted
                    if entity["entity"] in requested_dimensions]
        else:
            return extracted
