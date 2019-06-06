import pandas as pd
import os
import numpy as np

dirname = os.path.dirname(__file__)
embedding_dir = os.path.join(dirname, '../../data/embeddings/numberbatch')
glove_dir = os.path.join(dirname, '../../data/embeddings/glove')


def generate_numberbatch_csv(txt_path, csv_path):
    read = pd.read_csv(txt_path, sep=" ", skiprows=[
                       0], names=["id"] + [i for i in range(300)])
    read.to_csv(path_or_buf=csv_path, index=True)


def generate_glove_csv(txt_path, csv_path):
    read = pd.read_csv(txt_path, sep=" ", names=[
                       "id"] + [i for i in range(300)], quotechar=None, quoting=3)
    read.to_csv(path_or_buf=csv_path, index=True)


def generate_english_numberbatch_csv(in_csv_path, out_csv_path):
    df = pd.read_csv(in_csv_path)
    df = df[df['id'].str.contains('/c/en/')]
    df['id'] = df['id'].str.replace('/c/en/', '')
    df.to_csv(path_or_buf=out_csv_path, index=False)


def get_embeddings(df, words):
    ids = ['/c/en/' + word for word in words]
    embs = []
    for i in ids:
        row = df[df['id'].isin([i])].values
        if len(row) is 0:
            embs.append([i])
        else:
            embs.append(row[0])
    return {str(e[0]): np.array(e[1:]) if len(e) > 0 else [] for e in embs}


def get_embeddings_dataframe():
    return pd.read_csv(embedding_dir + '/numberbatch.csv')


def decapitate_csv(inp, out):
    with open(inp, 'r') as fin:
        data = [l.replace(",", " ") for l in fin.read().splitlines(True)]
    with open(out, 'w') as fout:
        fout.writelines(data[10:])


if __name__ == "__main__":
    # generate_numberbatch_csv(
    #     embedding_dir + '/numberbatch-en-17.06.txt', embedding_dir + '/numberbatch.csv')
    decapitate_csv(glove_dir + '/glove.6B.300d.txt',
                   glove_dir + '/glove.6B.300d.txt')
    generate_numberbatch_csv(
        glove_dir + '/glove.6B.300d.txt', glove_dir + '/glove.csv')
    # generate_english_numberbatch_csv(
    #     embedding_dir + '/numberbatch.csv', embedding_dir + '/numberbatch_eng.csv')
