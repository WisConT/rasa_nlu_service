import os
import time
from flair.data import Sentence
from flair.models import SequenceTagger

def eval_input(model, user_input):
    start_timestamp = time.time()
    sent = Sentence(user_input)
    model.predict(sent)
    result = sent.to_dict(tag_type='ner')
    end_timestamp = time.time()

    print("time taken: " + str(end_timestamp - start_timestamp))

    for entity in result['entities']:
        print(entity['type'] + ': ' + entity['text'] + ' (' + str(round(entity['confidence'], 3)) + ')')

    print()

def start():
    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, '../../../../models/flair/onto_uncased/glove_flair_embeddings/best-model.pt')
    model = SequenceTagger.load_from_file(model_path)

    print("Type your sentence...")
    user_input = input()

    while user_input != 'exit':
        eval_input(model, user_input)
        user_input = input()


if __name__ == '__main__':
    start()