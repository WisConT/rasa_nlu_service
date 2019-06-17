import os
import sys
import json

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../'))  # NOQA: E402

from utils import add_entities

# parses the onto notes data into a formatted array
def parse_file(filename):
    print("Parsing file...")

    with open(filename, 'r') as f:
        documents = []
        document = []

        sentence = {
            "words": [],
            "tags": [],
            "full_text": ""
        }

        for line in f:
            if line == "\n":
                if len(sentence['words']) > 0:
                    document.append(sentence)
                    sentence = {
                        "words": [],
                        "tags": [],
                        "full_text": ""
                    }

                continue

            split_line = line.split()

            if split_line[0] == '-DOCSTART-':
                if len(document) > 0:
                    documents.append(document)
                    document = []

                continue

            if len(sentence['words']) != 0:
                sentence['full_text'] = sentence['full_text'] + ' '

            sentence['words'].append(split_line[0])
            sentence['tags'].append(split_line[3])
            sentence['full_text'] = sentence['full_text'] + split_line[0]

        print("File parsed")
        
        documents = list(map(lambda doc: add_entities(doc), documents))

        return documents


def get_test_file():
    dirname = os.path.dirname(__file__)  # NOQA: E402
    test_file_path = os.path.join(
        dirname, '../../../data/processed/onto5/uncased/train.conll')
    
    return parse_file(test_file_path)


def get_entity_stats():
    test_file = get_test_file()

    meanings = {}

    for document in test_file:
        for sentence in document:
            for entity in sentence['entities']:
                if entity['value'] in meanings:
                    meanings[entity['value']].append(entity['entity'])
                else:
                    meanings[entity['value']] = [entity['entity']]
    
    total_meanings = 0

    for word, tags in meanings.items():
        unique_meanings = len(list(set(tags)))
        total_meanings += unique_meanings
    
    confusability = total_meanings / len(meanings.keys())

    print(confusability)


if __name__ == '__main__':
    get_entity_stats()
