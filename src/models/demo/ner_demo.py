import time
import os
import spacy
from termcolor import colored
from flair.data import Sentence
from flair.models import SequenceTagger
from prettytable import PrettyTable

def flair_eval(flair_model, user_input):
    start_timestamp = time.time()
    sent = Sentence(user_input)
    flair_model.predict(sent)
    result = sent.to_dict(tag_type='ner')
    end_timestamp = time.time()

    time_taken = round(end_timestamp - start_timestamp, 3)

    print()
    print("Flair results (" + str(time_taken) + "s):")

    table = PrettyTable(['Entity type', 'Text', 'Position', 'Confidence'])

    for entity in result['entities']:
        table.add_row([
            colored(entity['type'], attrs=['bold']),
            entity['text'],
            str(entity['start_pos']) + '-' + str(entity['end_pos']),
            round(entity['confidence'], 3)
        ])
        # print(colored(entity['type'], attrs=['bold']) + ': ' + entity['text'] + ' (' + str(round(entity['confidence'], 3)) + ')')

    table.align = 'l'
    table.align['Confidence'] = 'c'
    print(table)
    print()


def spacy_eval(spacy_model, user_input):
    start_timestamp = time.time()
    result = spacy_model(user_input)
    end_timestamp = time.time()

    time_taken = round(end_timestamp - start_timestamp, 3)

    print("spaCy results (" + str(time_taken) + "s):")

    table = PrettyTable(['Entity type', 'Text', 'Position'])

    for entity in result.ents:
        table.add_row([
            colored(entity.label_, attrs=['bold']),
            entity.text,
            str(entity.start_char) + '-' + str(entity.end_char)
        ])
        # print(colored(entity.label_, attrs=['bold']) + ': ' + entity.text)

    table.align = 'l'
    print(table)
    print()

def run_chatbot():
    dirname = os.path.dirname(__file__)
    flair_model_path = os.path.join(dirname, '../../../models/flair/onto_uncased/glove_flair_embeddings/best-model.pt')
    flair_model = SequenceTagger.load_from_file(flair_model_path)

    spacy_model_path = os.path.join(dirname, '../../../models/spacy/onto_uncased/model-best')
    spacy_model = spacy.load(spacy_model_path)
    
    print()
    print("Type your input:")

    while True:
        print("> ", end = "")
        user_input = input()

        if user_input == 'exit':
            break

        flair_eval(flair_model, user_input)
        spacy_eval(spacy_model, user_input)
    
    print("Stopping...")

if __name__ == '__main__':
    run_chatbot()