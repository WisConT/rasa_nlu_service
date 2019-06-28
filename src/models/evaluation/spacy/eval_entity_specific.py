import sys
import os
import json
import spacy
import pandas as pd
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../../'))  # NOQA: E402

from data.onto.make_dataset import get_dataset
from baseline_model import get_statistics, entities_equal


def calculate_test_results(model_path, results_path, cased=True):
    test, train, dev = get_dataset(cased=cased)

    nlp = spacy.load(model_path)

    statistics = get_statistics(test, nlp)

    print("statistics: ")
    print(json.dumps(statistics, indent=4))

    # results_file = os.path.join(dirname, results_path)
    # f = open(results_file, "w+")
    # f.write(json.dumps(statistics, indent=4))
    # f.close()


def get_formatted_data(results_path):
    dirname = os.path.dirname(__file__)
    results_file = os.path.join(dirname, results_path)

    entity_map = {
        "fac": "Facility",
        "work_of_art": "Work of art",
        "product": "Product",
        "event": "Event",
        "loc": "Location",
        "law": "Law",
        "time": "Time",
        "language": "Language",
        "quantity": "Quantity",
        "org": "Organisation",
        "ordinal": "Ordinal",
        "person": "Person",
        "cardinal": "Cardinal",
        "norp": "NORP",
        "date": "Date",
        "gpe": "GPE",
        "money": "Money",
        "percent": "Percent"
    }

    with open(results_file, 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data['performance']).T
        df = df.drop(['4', '5'], axis=0)
        df = df.rename(entity_map, axis='index')

        return df


def evaluate_results(data_path):
    df = get_formatted_data(data_path)

    avg = df['f1']['corpus_average']
    df_f1 = df['f1'].drop('corpus_average').sort_values()

    df_f1 = df_f1 * 100

    print("corpus total f1: " + str(avg))

    ax1 = df_f1.plot.bar()
    # ax1.axhline(y=avg_uncased, color='royalblue', linestyle='--', zorder=0.01, linewidth=1)
    # ax1.axhline(y=avg_cased, color='orangered', linestyle='--', zorder=0.01, linewidth=1)
    ax1.set(xlabel="Entity", ylabel="F1 score")
    ax1.grid(color='grey', linestyle='--', linewidth=0.5, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.show()


def evaluate_results_dual():
    df_uncased_model = get_formatted_data("results/onto_uncased_model_cased_data.json")
    df_cased_model = get_formatted_data("results/onto_cased_model_uncased_data.json")

    avg_uncased = df_uncased_model['f1']['corpus_average']
    df_uncased_f1 = df_uncased_model['f1'].drop('corpus_average')

    avg_cased = df_cased_model['f1']['corpus_average']
    df_cased_f1 = df_cased_model['f1'].drop('corpus_average')

    print("corpus total (uncased model): " + str(avg_uncased))
    print("corpus total (cased model): " + str(avg_cased))

    joined_df = pd.DataFrame({
        "Train: Uncased - Test: Cased": df_uncased_f1 * 100,
        "Train: Cased - Test: Uncased": df_cased_f1 * 100
    }).sort_values('Train: Uncased - Test: Cased')

    ax1 = joined_df.plot.bar()
    # ax1.axhline(y=avg_uncased, color='royalblue', linestyle='--', zorder=0.01, linewidth=1)
    # ax1.axhline(y=avg_cased, color='orangered', linestyle='--', zorder=0.01, linewidth=1)
    ax1.set(xlabel="Entity", ylabel="F1 score")
    ax1.grid(color='grey', linestyle='--', linewidth=0.5, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    uncased_path = os.path.join(dirname, '../../../../models/spacy/onto_uncased/model-best')
    cased_path = os.path.join(dirname, '../../../../models/spacy/onto_cased/model-best')

    # calculate_test_results('en_core_web_lg', 'results/onto_uncased_spacy_en_core_web_lg.json', False)
    # evaluate_results('results/onto_uncased_spacy_en_core_web_sm.json')
    # calculate_test_results(uncased_path, "results/onto_uncased_spacy_en_core_web_lg.json", False)
    # calculate_test_results(cased_path, "results/onto_cased_entity_specific.json", True)
    evaluate_results_dual()