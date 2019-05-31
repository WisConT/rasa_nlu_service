import os
import pandas as pd
import matplotlib.pyplot as plt
import json

def get_loss_data(path):
    dirname = os.path.dirname(__file__)  # NOQA: E402
    model_path = os.path.join(dirname, path)

    training_stats = []

    for i in range(50):
        model_info_path = os.path.join(model_path, 'model' + str(i) + '/accuracy.json')
        with open(model_info_path, 'r') as f:
            training_stats.append(json.load(f))

    return pd.DataFrame(training_stats)

def evaluate():
    df_cased = get_loss_data('../../../../models/spacy/onto_cased')
    df_uncased = get_loss_data('../../../../models/spacy/onto_uncased')

    cased_f_score = df_cased['ents_f']
    uncased_f_score = df_uncased['ents_f']

    ax = cased_f_score.plot(title='F1 score at each epoch')
    uncased_f_score.plot(ax=ax)
    ax.set(xlabel="Epochs", ylabel="F1 score")
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.legend(["Cased", "Uncased"])

    plt.show()


if __name__ == '__main__':
    evaluate()