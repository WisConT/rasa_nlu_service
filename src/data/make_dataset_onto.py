import os
from data.make_dataset import add_entities
from data.clean_dataset_onto import clean_dataset

# parses the onto files into a formatted array


def parse_file(filename):
    with open(filename, 'r', encoding="ISO-8859-1") as f:
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


def get_dataset():
    print("Fetching dataset...")

    dirname = os.path.dirname(__file__)  # NOQA: E402
    interim_data = os.path.join(
        dirname, '../../data/interim/onto5/english/annotations/')
    directory_paths = [(dirpath, filenames)
                       for (dirpath, dirnames, filenames) in os.walk(interim_data)]

    documents = []

    for (dirname, files) in directory_paths:
        for ner_file in files:
            file_path = os.path.join(dirname, ner_file)
            document = parse_file(file_path)
            documents.append(document)

    return documents
