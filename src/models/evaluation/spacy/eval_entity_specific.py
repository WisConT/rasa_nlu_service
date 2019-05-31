import sys
import os
import json
import spacy
import pandas as pd
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../../'))  # NOQA: E402

from data.onto.make_dataset import get_dataset
from baseline_model import get_statistics, entities_equal


def calculate_test_results(model_path, results_path, cased=True):
    test, train, dev = get_dataset(cased=cased)

    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, model_path)

    nlp = spacy.load(model_path)

    statistics = get_statistics(test, nlp)

    print("statistics: ")
    print(json.dumps(statistics, indent=4))

    results_file = os.path.join(dirname, results_path)
    f = open(results_file, "w+")
    f.write(json.dumps(statistics, indent=4))
    f.close()


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


def evaluate_results():
    df_uncased = get_formatted_data("results/onto_uncased_entity_specific.json")
    df_cased = get_formatted_data("results/onto_cased_entity_specific.json")

    avg_uncased = df_uncased['f1']['corpus_average']
    df_uncased_f1 = df_uncased['f1'].drop('corpus_average')

    avg_cased = df_cased['f1']['corpus_average']
    df_cased_f1 = df_cased['f1'].drop('corpus_average')

    print("corpus avg (uncased): " + str(avg_uncased))
    print("corpus avg (cased): " + str(avg_cased))

    joined_df = pd.DataFrame({
        "uncased": df_uncased_f1,
        "cased": df_cased_f1
    }).sort_values('cased')

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
    # calculate_test_results('../../../../models/spacy/onto_uncased/model-best', "results/onto_uncased_entity_specific.json", False)
    # calculate_test_results('../../../../models/spacy/onto_cased/model-best', "results/onto_cased_entity_specific.json", True)
    evaluate_results()