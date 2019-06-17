from functools import reduce
import os
import time
from flair.data import Sentence


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

    if pred['type'].lower() in mappings:
        name_equal = true['entity'].lower() in mappings[pred['type'].lower()]
    else:
        name_equal = pred['type'].lower() == true['entity'].lower()

    value_equal = pred['text'] == true['value']
    value_similar = pred['text'] in true['value'] or true['value'] in pred['text']

    range_equal = pred['start_pos'] == true['start'] and pred['end_pos'] == true['end']
    range_similar = \
        pred['start_pos'] <= true['start'] and pred['end_pos'] >= true['end'] \
        or \
        true['start'] <= pred['start_pos'] and true['end'] >= pred['end_pos']

    return (name_equal and value_equal and range_equal) or (name_equal and value_similar and range_similar)


def flair_spacy_entity_equal(flair_entity, spacy_entity):
    name_equal = flair_entity['type'].lower() == spacy_entity.label_.lower()

    value_equal = flair_entity['text'] == spacy_entity.text
    value_similar = flair_entity['text'] in spacy_entity.text or spacy_entity.text in flair_entity['text']

    range_equal = flair_entity['start_pos'] == spacy_entity.start_char and flair_entity['end_pos'] == spacy_entity.end_char
    range_similar = \
        flair_entity['start_pos'] <= spacy_entity.start_char and flair_entity['end_pos'] >= spacy_entity.end_char \
        or \
        spacy_entity.start_char <= flair_entity['start_pos'] and spacy_entity.end_char >= flair_entity['end_pos']

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


def get_statistics(documents, flair_model, mappings={}, spacy_labels=None):
    """Given a list of documents in the format given from parse_file functions
    get the recall, precision and f1 scores for the documents.

    Parameters:
        documents: list of documents provided by the parse_file function
        interpreter: the RASA interpreter to perform NER

    """
    print("Calculating corpus statistics...")

    statistics = {}
    entity_meanings = {}

    document_count = 0
    sentence_count = 0
    words_count = 0
    entities_count = 0

    entity_reference_count = {}

    eval_time = 0
    eval_count = 0
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

            eval_start_timestamp = time.time()
            sent = Sentence(sentence['full_text'])
            flair_model.predict(sent)
            result = sent.to_dict(tag_type='ner')
            eval_end_timestamp = time.time()

            # spacy_result = spacy_model(sentence['full_text'])

            eval_time += eval_end_timestamp - eval_start_timestamp
            eval_count += 1

            for predicted_entity in result['entities']:
                if spacy_labels is not None and predicted_entity['type'].lower() not in spacy_labels:
                    continue
                # spacy_found = False

                # for spacy_entity in spacy_result.ents:
                #     if flair_spacy_entity_equal(predicted_entity, spacy_entity):
                #         spacy_found = True
                
                # if not spacy_found:
                #     print("SPACY NOT FOUND:")
                #     print(sentence['full_text'])
                #     print(predicted_entity)


                found = False
                entity_name = predicted_entity['type']

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

                if spacy_labels is not None and true_entity['entity'].lower() not in spacy_labels:
                    continue

                found = False

                for predicted_entity in result['entities']:
                    if entities_equal(predicted_entity, true_entity, mappings):
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
        # "avg_entities_per_word": avg_entities_per_word
    }

    return {
        "performance": perf_stats,
        "meta": meta_stats,
        "entity_stats": entity_stats,
        "time_taken": end_timestamp - start_timestamp,
        "average_eval_time": eval_time / eval_count
    }
