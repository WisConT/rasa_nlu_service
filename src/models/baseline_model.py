from rasa_nlu.model import Interpreter
import os

def entities_equal(pred, true, mappings):
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

def get_statistics(documents, spacy_labels, mappings):
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
                if predicted_entity['entity'].lower() not in spacy_labels:
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
                if predicted_entity['entity'].lower() not in spacy_labels:
                    continue

                found = False

                for predicted_entity in result['entities']:
                    if entities_equal(predicted_entity, true_entity, mappings):
                        found = True
                        break
                
                if not found:
                    false_negative = false_negative + 1
    
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return (recall, precision, f1)
