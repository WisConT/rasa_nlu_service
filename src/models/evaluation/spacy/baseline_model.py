from functools import reduce
import os
import time


def entities_equal(pred, true):
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

    name_equal = pred.label_.lower() == true['entity'].lower()

    value_equal = pred.text == true['value']
    value_similar = pred.text in true['value'] or true['value'] in pred.text

    range_equal = pred.start_char == true['start'] and pred.end_char == true['end']
    range_similar = \
        pred.start_char <= true['start'] and pred.end_char >= true['end'] \
        or \
        true['start'] <= pred.start_char and true['end'] >= pred.end_char

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
    statistics['corpus_average'] = {
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
            entity_stats['f1'] = (2 * entity_stats['precision'] * entity_stats['recall']) / pr_re
        else:
            entity_stats['f1'] = 0
    
    return statistics


def get_statistics(documents, nlp):
    """Given a list of documents in the format given from parse_file functions
    get the recall, precision and f1 scores for the documents.

    Parameters:
        documents: list of documents provided by the parse_file function
        interpreter: the RASA interpreter to perform NER

    Returns:
        (recall, precision, f1)

    """
    print("Calculating corpus statistics...")

    statistics = {}

    document_count = 0
    sentence_count = 0
    words_count = 0
    entities_count = 0

    entity_reference_count = {}

    start_timestamp = time.time()

    if len(documents) == 0:
        print("Documents is empty...")
        return

    while document_count < len(documents):
        print("doc: " + str(document_count + 1) + "/" + str(len(documents)))
        document = documents[document_count]

        for sentence in document:
            sentence_count += 1
            words_count += len(sentence['words'])

            result = nlp(sentence['full_text'])

            for predicted_entity in result.ents:
                found = False
                entity_name = predicted_entity.label_

                for true_entity in sentence['entities']:
                    if entities_equal(predicted_entity, true_entity):
                        found = True
                        entity_name = true_entity['entity']
                        break

                if found:
                    statistics = increment_field(statistics, entity_name.lower(), 'true_positive')
                else:
                    statistics = increment_field(statistics, entity_name.lower(), 'false_positive')

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

                found = False

                for predicted_entity in result.ents:
                    if entities_equal(predicted_entity, true_entity):
                        found = True
                        break

                if not found:
                    statistics = increment_field(statistics, true_entity_name, 'false_negative')

        document_count += 1
    
    end_timestamp = time.time()

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
        "entity_stats": entity_stats,
        "time_taken": end_timestamp - start_timestamp
    }
