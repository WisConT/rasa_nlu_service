import pandas as pd
import os
import json
import matplotlib.pyplot as plt

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
    # df_glove_flair = get_formatted_data("results/onto_uncased_glove_flair.json")
    # df_glove_flair = get_formatted_data("results/onto_uncased_.json")
    # df_spacy = get_formatted_data("../spacy/results/onto_uncased_spacy_en_core_web_lg.json")

    # avg_flair = df_flair['f1']['corpus_average']
    # df_flair_f1 = df_flair['f1'].drop('corpus_average')

    # avg_spacy = df_spacy['f1']['corpus_average']
    # df_spacy_f1 = df_spacy['f1'].drop('corpus_average')

    # print("corpus total (flair): " + str(avg_flair))
    # print("corpus total (spacy): " + str(avg_spacy))

    df = pd.DataFrame({
        'lab': ['spaCy - Custom Trained', 'Flair-Fast', 'Flair-Crawl'],
        'val': [7.61, 69.73, 142.46]
    })

    ax1 = df.plot.bar(x='lab', y='val')
    # ax1.axhline(y=avg_uncased, color='royalblue', linestyle='--', zorder=0.01, linewidth=1)
    # ax1.axhline(y=avg_cased, color='orangered', linestyle='--', zorder=0.01, linewidth=1)
    ax1.set(xlabel="Model", ylabel="Training time (hours)")
    ax1.grid(color='grey', linestyle='--', linewidth=0.5, axis='y')
    ax1.legend([])

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_results()