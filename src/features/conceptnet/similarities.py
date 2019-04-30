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


class ConceptNetSimilarity(Component):

    name = "similarities"
    provides = ["entities"]
    requires = ["tokens"]
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config=None):
        print("found component")
        super(ConceptNetSimilarity, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        """Load the sentiment polarity labels from the text
           file, retrieve training tokens and after formatting
           data train the classifier."""

        # with open('labels.txt', 'r') as f:
        #    labels = f.read().splitlines()

        training_data = training_data.training_examples  # list of Message objects
        print([m.data for m in training_data[0:1]])

        tokens = [list(map(lambda x: x.text, t.get('tokens')))
                  for t in training_data]
        processed_tokens = [self.preprocessing(t) for t in tokens]
        labeled_data = [(t, x) for t, x in zip(processed_tokens, labels)]
        self.clf = NaiveBayesClassifier.train(labeled_data)

    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "",  # TODO
                  "extractor": self.name}

        return entity

    def preprocessing(self, tokens):
        print(tokens)
        """
        Given tokens, create embeddings
        """
        df = nb.get_embeddings_dataframe()
        classes = ["person", "geopolitical_entity"]
        classes_emb = nb.get_embeddings(df, classes)

        # need to make sure that there is an embedding
        tokens_emb = nb.get_embeddings(df, tokens)
        tokens_emb = cn.add_features_to_embeddings(
            df, tokens_emb, classes_emb)

        return tokens_emb

    def process(self, message, **kwargs):
        """Retrieve the tokens of the new message, pass it to the classifier
            and append prediction results to the message class."""

        if not self.clf:
            # component is either not trained or didn't
            # receive enough training data
            entity = None
        else:
            tokens = [t.text for t in message.get("tokens")]
            tb = self.preprocessing(tokens)
            pred = self.clf.prob_classify(tb)

            sentiment = pred.max()
            confidence = pred.prob(sentiment)

            entity = self.convert_to_rasa(sentiment, confidence)

            message.set("entities", [entity], add_to_output=True)

    def persist(self, model_dir):
        """Persist this model into the passed directory."""

        classifier_file = os.path.join(model_dir, SIMILARITY_MODEL_FILE_NAME)
        utils.pycloud_pickle(classifier_file, self)
        return {"classifier_file": SIMILARITY_MODEL_FILE_NAME}

    @classmethod
    def load(cls,
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs):

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("classifier_file", SIMILARITY_MODEL_FILE_NAME)
        classifier_file = os.path.join(model_dir, file_name)

        if os.path.exists(classifier_file):
            return utils.pycloud_unpickle(classifier_file)
        else:
            return cls(meta)


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
        self.nlp = spacy.load('./xyz')
        super(SpacyNBEntityExtractor, self).__init__(component_config)

    def process(self, message: Message, **kwargs: Any) -> None:
        # can't use the existing doc here (spacy_doc on the message)
        # because tokens are lower cased which is bad for NER
        # spacy_nlp = kwargs.get("spacy_nlp", None)
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
