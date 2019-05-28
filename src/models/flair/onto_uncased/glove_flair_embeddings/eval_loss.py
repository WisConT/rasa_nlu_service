import os
import pandas as pd
import matplotlib.pyplot as plt

def get_loss_data():
    dirname = os.path.dirname(__file__)  # NOQA: E402
    data_path = os.path.join(dirname, '../../../../../models/flair/onto_uncased/loss.tsv')

    return pd.read_csv(data_path, sep="\t")

def evaluate():
    df = get_loss_data()

    x = len(df)
    num_index = range(0,x,1)
    df['EPOCH_DATE'] = df.index
    df =  df.reset_index()

    print(df)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('dev. loss', color=color)
    ax1.plot(df['index'], df['DEV_LOSS'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('f1 score', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['index'], df['TEST_F-SCORE'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == '__main__':
    evaluate()