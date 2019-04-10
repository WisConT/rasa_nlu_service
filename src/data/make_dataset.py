# adds a spaCy-style entity list to the sentence dictionaries
def add_entities(document):
    for sentence in document:
        # create entities list
        sentence['entities'] = []

        words = sentence['words']
        tags = sentence['tags']
        count = 0

        i = 0
        while i < len(words):
            word = words[i]
            tag = tags[i]

            # if its start of a tag
            if tag[0] == 'B':
                entity = {
                    "entity": tag.strip('B-'),
                    "value": word,
                    "start": count,
                    "end": count + len(word)
                }

                # count keeps track of index in the string
                count = count + (0 if i == len(words) -
                                    1 else 1) + len(word)

                # find any further tags that relate to the current tag
                # (start with I)
                j = i + 1
                while j < len(words) and (tags[j][0] == 'I' or words[j] == "'s"):
                    # update the value of the entity to include next word
                    entity['value'] = entity['value'] + ' ' + words[j]

                    # update the end index of the entity (including space)
                    entity['end'] = entity['end'] + 1 + len(words[j])

                    # count keeps track of index in the string
                    count = count + (0 if i == len(words) -
                                        1 else 1) + len(words[j])
                    j = j + 1

                # skip over the tags weve just parsed
                i = j

                # add the full entity to the list
                sentence['entities'].append(entity)
            else:
                # count keeps track of index in the string
                count = count + (0 if i == len(words) -
                                    1 else 1) + len(word)

                # no entity here, continue
                i = i + 1

    return document
