import os
import sys

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../'))  # NOQA: E402

from utils import add_entities

# parses the onto files into a formatted array


def parse_file(filename):
    with open(filename, 'r') as f:
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

            if len(sentence['words']) != 0:
                sentence['full_text'] = sentence['full_text'] + ' '

            sentence['words'].append(split_line[0])
            sentence['tags'].append(split_line[1])
            sentence['full_text'] = sentence['full_text'] + split_line[0]

        return add_entities(document)


def get_dataset(cased=True):
    print("Fetching dataset...")

    dirname = os.path.dirname(__file__)  # NOQA: E402
    data_path = '../../../data/interim/onto5/data_cased.iob2' if cased else '../../../data/interim/onto5/data_uncased.iob2'
    data_file = os.path.join(dirname, data_path)

    document = parse_file(data_file)

    return [document]
