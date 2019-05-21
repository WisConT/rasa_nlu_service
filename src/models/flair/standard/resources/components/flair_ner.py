from rasa_nlu.components import Component
from rasa_nlu.config import override_defaults

from flair.data import Sentence
from flair.models import SequenceTagger

class FlairNER(Component):

    name = "flair_ner"
    provides = ["entities"]
    requires = ["tokens"]
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config=None):
        self.tagger = SequenceTagger.load('ner')
        super(FlairNER, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        pass

    def preprocessing(self, tokens):
        pass

    def convert_to_rasa(self, entity):
        return {
            "value": entity['text'],
            "confidence": entity['confidence'],
            "entity": entity['type'],
            "start": entity['start_pos'],
            "end": entity['end_pos'],
            "extractor": "flair_ner"
        }

    def process(self, message, **kwargs):
        sentence = Sentence(message.text)
        self.tagger.predict(sentence)

        sent = sentence.to_dict(tag_type='ner')

        formatted_entities = list(map(lambda e: self.convert_to_rasa(e), sent['entities']))

        message.set("entities", formatted_entities, add_to_output=True)

    def persist(self, model_dir):
        pass

