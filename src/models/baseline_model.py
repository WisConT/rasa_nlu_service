from rasa_nlu.model import Interpreter
import os


def entities_equal(pred, true, mappings={}):
    """Check if two entities are equal, also takes an optional mappings dict
    which maps from a spacy entity name to all possible alternatives in the
    data set

    NOTE:
        - ensure pred is the spacy predicted entity
        - ensure the mapping is a dictionary of (lowercase) spacy entity names
          to arrays of possible alternative entity names in the target data set
          e.g.:

            mappings = {
                "gpe": ["location"],
                "org": ["group", "corporation"]
            }

    Parameters:
        pred: the spacy predicted entity
        true: the ground truth entity from the data set
        mappings: a dictionary of mappings from spacy entity types to all
            possible alternatives in the data set (NOTE: lowercase)

    Returns:
        True/False
    """
    if pred['entity'].lower() in mappings:
        name_equal = true['entity'].lower() in mappings[pred['entity'].lower()]
    else:
        name_equal = pred['entity'].lower() == true['entity'].lower()

    value_equal = pred['value'] == true['value']
    value_similar = pred['value'] in true['value'] or true['value'] in pred['value']

    range_equal = pred['start'] == true['start'] and pred['end'] == true['end']
    range_similar = \
        pred['start'] <= true['start'] and pred['end'] >= true['end'] \
        or \
        true['start'] <= pred['start'] and true['end'] >= pred['end']

    return (name_equal and value_equal and range_equal) or (name_equal and value_similar and range_similar)


def get_statistics(documents, spacy_labels=None, mappings={}):
    """Given a list of documents in the format given from parse_file functions
    get the recall, precision and f1 scores for the documents.

    Parameters:
        documents: list of documents provided by the parse_file function
        spacy_labels: list of all the spacy entity names to be used, entities
            that are not in this list will be ignored, do not pass a value and
            all labels will be used (NOTE: lowercase)
        mappings: a dictionary of mappings from spacy entity types to all
            possible alternatives in the data set (NOTE: lowercase)

    Returns:
        (recall, precision, f1)

    """
    print("Calculating corpus statistics...")

    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, '../../models/nlu/default/current')
    interpreter = Interpreter.load(model_path)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(documents)):
        print("doc: " + str(i + 1) + "/" + str(len(documents)))
        document = documents[i]

        for sentence in document:
            result = interpreter.parse(sentence['full_text'])

            for predicted_entity in result['entities']:
                if spacy_labels is not None and predicted_entity['entity'].lower() not in spacy_labels:
                    continue

                found = False

                for true_entity in sentence['entities']:
                    if entities_equal(predicted_entity, true_entity, mappings):
                        found = True

                if found:
                    true_positive = true_positive + 1
                else:
                    false_positive = false_positive + 1

            for true_entity in sentence['entities']:
                if spacy_labels is not None and predicted_entity['entity'].lower() not in spacy_labels:
                    continue

                found = False

                for predicted_entity in result['entities']:
                    if entities_equal(predicted_entity, true_entity, mappings):
                        found = True
                        break

                if not found:
                    false_negative = false_negative + 1

    recall = true_positive / \
        (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    precision = true_positive / \
        (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    f1 = (2 * precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    return (recall, precision, f1)
