import os
import pandas as pd
import matplotlib.pyplot as plt
import json

def get_loss_data(path):
    dirname = os.path.dirname(__file__)  # NOQA: E402
    model_path = os.path.join(dirname, path)

    training_stats = []

    i = 0
    dirs = os.listdir(model_path)

    while ('model' + str(i)) in dirs:
        model_info_path = os.path.join(model_path, 'model' + str(i) + '/accuracy.json')
        with open(model_info_path, 'r') as f:
            training_stats.append(json.load(f))
        i += 1

    return pd.DataFrame(training_stats)

def evaluate():
    df_cased = get_loss_data('../../../../models/spacy/onto_cased')
    df_uncased = get_loss_data('../../../../models/spacy/onto_uncased')

    uncased_f_score = df_uncased['ents_f']
    cased_f_score = df_cased['ents_f']

    ax = uncased_f_score.plot(title='Dev set F1 score at each epoch')
    cased_f_score.plot(ax=ax)
    ax.set(xlabel="Epochs", ylabel="F1 score")
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.legend(["Uncased", "Cased"])

    plt.show()


if __name__ == '__main__':
    evaluate()