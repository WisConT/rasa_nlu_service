from data.make_dataset import add_entities

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

        # 1 line = 1 word
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

        return add_entities(documents)
