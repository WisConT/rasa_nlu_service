import sys
import os
import json
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from flair.models import SequenceTagger

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../../'))  # NOQA: E402

from data.onto.make_dataset import get_dataset
from baseline_model import get_statistics, entities_equal


def calculate_test_results(model_path, results_path, cased=True):
    test, train, dev = get_dataset(cased=cased)

    flair_model = SequenceTagger.load_from_file(model_path)

    statistics = get_statistics(test, flair_model)

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
    df_flair = get_formatted_data("results/onto_uncased_glove_flair.json")
    df_spacy = get_formatted_data("../spacy/results/onto_uncased_spacy_en_core_web_lg.json")
    # df_spacy_sm = get_formatted_data("../spacy/results/onto_uncased_spacy_en_core_web_sm.json")

    avg_flair = df_flair['f1']['corpus_average']
    df_flair_f1 = df_flair['f1'].drop('corpus_average')

    avg_spacy = df_spacy['f1']['corpus_average']
    df_spacy_f1 = df_spacy['f1'].drop('corpus_average')

    # df_spacy_sm_f1 = df_spacy_sm['f1'].drop('corpus_average')

    print("corpus total (flair): " + str(avg_flair))
    print("corpus total (spacy): " + str(avg_spacy))

    joined_df = pd.DataFrame({
        "spaCy (custom)": df_spacy_f1 * 100,
        "Flair-Crawl": df_flair_f1 * 100
    }).sort_values('Flair-Crawl')

    # ax1 = joined_df.plot.bar(color=['#1f76b4', 'purple', '#ff7f0f'])
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
    model_path = os.path.join(dirname, '../../../../models/flair/onto_uncased/glove_flair_embeddings/best-model.pt')

    # calculate_test_results(model_path, "results/onto_uncased_flair_fast.json", True)
    # calculate_test_results(model_path, "results/onto_uncased_glove_flair.json", False)
    # evaluate_results('results/onto_uncased_glove_flair.json')
    evaluate_results_dual()
