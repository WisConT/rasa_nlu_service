import os
from utils import add_entities

# parses the conll file into a formatted array


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


def get_dataset():
    print("Fetching dataset...")

    dirname = os.path.dirname(__file__)  # NOQA: E402

    return parse_file(os.path.join(
        dirname, '../../data/interim/conll_2003/test.txt'))
