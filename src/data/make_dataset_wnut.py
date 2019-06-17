import os
import sys

# parses the conll file into a formatted array
dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '.'))  # NOQA: E402

from utils import add_entities

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

        # 1 line = 1 word
        for line in f:
            if line == "\n" or len(line.split()) == 0:
                if len(sentence['words']) > 0:
                    document.append(sentence)
                    sentence = {
                        "words": [],
                        "tags": [],
                        "full_text": ""
                    }

                continue

            split_line = line.split()

            # add a space before each new word
            if len(sentence['words']) > 0:
                sentence['full_text'] = sentence['full_text'] + ' '

            sentence['words'].append(split_line[0])
            sentence['tags'].append(split_line[1])
            sentence['full_text'] = sentence['full_text'] + split_line[0]

            if len(document) > 0:
                documents.append(document)

            document = []

        print("File parsed")

        # remove all sentences that contain a @mention
        # documents = list(filter(lambda d: "@" not in d[0]['words'], documents))

        documents = list(map(lambda doc: add_entities(doc), documents))

        return documents


def get_dataset():
    print("Fetching dataset...")

    dirname = os.path.dirname(__file__)  # NOQA: E402

    test = parse_file(os.path.join(
        dirname, '../../data/interim/wnut_2017/test.txt'))
    train = parse_file(os.path.join(
        dirname, '../../data/interim/wnut_2017/train.txt'))

    return test + train
