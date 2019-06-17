from rasa_nlu.components import Component
from rasa_nlu.config import override_defaults

from flair.data import Sentence
from flair.models import SequenceTagger

import os

class FlairNER(Component):

    name = "flair_ner"
    provides = ["entities"]
    defaults = {
        # by default all dimensions recognized by spacy are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None
    }
    language_list = ["en"]

    def __init__(self, component_config = None):
        super(FlairNER, self).__init__(component_config)

        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, '../../../../../../../models/flair/onto_uncased/glove_flair_embeddings/best-model.pt')
        self.model = SequenceTagger.load_from_file(model_path)

    def process(self, message, **kwargs):
        sent = Sentence(message.text)
        self.model.predict(sent)
        result = sent.to_dict(tag_type='ner')
        all_extracted = self.extract_entities(result)
        message.set(
            "entities", message.get("entities", []) + all_extracted, add_to_output=True
        )

    @staticmethod
    def extract_entities(result):
        entities = [
            {
                "entity": ent['type'],
                "value": ent['text'],
                "start": ent['start_pos'],
                "confidence": ent['confidence'],
                "end": ent['end_pos'],
            }
            for ent in result['entities']
        ]
        return entities
