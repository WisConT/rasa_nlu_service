from data.make_dataset import add_entities

# parses the conll file into a formatted array
def parse_file(filename='test.txt'):
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
        return add_entities(documents)
