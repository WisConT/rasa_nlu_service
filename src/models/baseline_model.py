from rasa_nlu.model import Interpreter
from functools import reduce
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


def increment_field(statistics, entity, field):
    if entity not in statistics:
        statistics[entity] = {
            'true_positive': 0,
            'false_positive': 0,
            'false_negative': 0
        }

    statistics[entity][field] += 1

    return statistics


def calculate_performance_statistics(statistics):
    # calculate performance stats for total corpus
    statistics['corpus_total'] = {
        'true_positive': reduce(lambda x, v: x + v['true_positive'], statistics.values(), 0),
        'false_positive': reduce(lambda x, v: x + v['false_positive'], statistics.values(), 0),
        'false_negative': reduce(lambda x, v: x + v['false_negative'], statistics.values(), 0)
    }

    for entity in statistics:
        entity_stats = statistics[entity]

        tp_fn = entity_stats['true_positive'] + entity_stats['false_negative']
        tp_fp = entity_stats['true_positive'] + entity_stats['false_positive']

        if tp_fn > 0:
            entity_stats['recall'] = entity_stats['true_positive'] / tp_fn
        else:
            entity_stats['recall'] = 0

        if tp_fp > 0:
            entity_stats['precision'] = entity_stats['true_positive'] / tp_fp
        else:
            entity_stats['precision'] = 0

        pr_re = entity_stats['precision'] + entity_stats['recall']

        if pr_re > 0:
            entity_stats['f1'] = (
                2 * entity_stats['precision'] * entity_stats['recall']) / pr_re
        else:
            entity_stats['f1'] = 0

    return statistics


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

    statistics = {}

    document_count = 0
    sentence_count = 0
    words_count = 0
    entities_count = 0

    entity_reference_count = {}

    while document_count < len(documents):
        print("doc: " + str(document_count + 1) + "/" + str(len(documents)))
        document = documents[document_count]

        for sentence in document:
            sentence_count += 1
            words_count += len(sentence['words'])

            result = interpreter.parse(sentence['full_text'])

            for predicted_entity in result['entities']:
                if spacy_labels is not None and predicted_entity['entity'].lower() not in spacy_labels:
                    continue

                found = False
                entity_name = predicted_entity['entity']

                for true_entity in sentence['entities']:
                    if entities_equal(predicted_entity, true_entity, mappings):
                        found = True
                        entity_name = true_entity['entity']
                        break

                # if there is only one possible mapping, use that mapping for
                # statistics
                if entity_name.lower() in mappings and len(mappings[entity_name.lower()]) == 1:
                    entity_name = mappings[entity_name.lower()][0]

                if found:
                    statistics = increment_field(
                        statistics, entity_name.lower(), 'true_positive')
                else:
                    statistics = increment_field(
                        statistics, entity_name.lower(), 'false_positive')

            for true_entity in sentence['entities']:
                entities_count += 1
                true_entity_name = true_entity['entity'].lower()
                true_entity_value = true_entity['value'].lower()

                # collect stats about avg. number of entities a given words referrs to
                if true_entity_value in entity_reference_count:
                    if true_entity_name not in entity_reference_count[true_entity_value]:
                        entity_reference_count[true_entity_value].append(true_entity_name)
                else:
                    entity_reference_count[true_entity_value] = [true_entity_name]

                if spacy_labels is not None and predicted_entity['entity'].lower() not in spacy_labels:
                    continue

                found = False

                for predicted_entity in result['entities']:
                    if entities_equal(predicted_entity, true_entity, mappings):
                        found = True
                        break

                if not found:
                    statistics = increment_field(
                        statistics, true_entity_name, 'false_negative')

        document_count += 1

    total_number_of_entities = sum(list(map(lambda x: len(x), entity_reference_count.values())))
    
    # average number of unique entities a phrase/word refers to, ignoring those
    # that refer to no entities
    avg_entities_per_word = total_number_of_entities / len(entity_reference_count)

    # calculate precision, recall, f1
    perf_stats = calculate_performance_statistics(statistics)
    meta_stats = {
        "document_count": document_count,
        "sentence_count": sentence_count,
        "words_count": words_count,
        "entities_count": entities_count
    }
    entity_stats = {
        "avg_entities_per_word": avg_entities_per_word
    }

    return {
        "performance": perf_stats,
        "meta": meta_stats,
        "entity_stats": entity_stats
    }
