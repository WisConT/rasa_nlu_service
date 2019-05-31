import os
import sys

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

def get_dataset(cased=True):
    print("Fetching dataset...")

    cased_path = 'cased' if cased else 'uncased'

    dirname = os.path.dirname(__file__)  # NOQA: E402
    test_file_path = os.path.join(
        dirname, '../../../data/processed/onto5/' + cased_path + '/test.conll')
    train_file_path = os.path.join(
        dirname, '../../../data/processed/onto5/' + cased_path + '/train.conll')
    dev_file_path = os.path.join(
        dirname, '../../../data/processed/onto5/' + cased_path + '/dev.conll')

    test = parse_file(test_file_path)
    train = parse_file(train_file_path)
    dev = parse_file(dev_file_path)

    return test, train, dev
