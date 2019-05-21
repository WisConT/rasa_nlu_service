from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
import os

def train_model():
    columns = {0: 'text', 1: 'ner'}

    dirname = os.path.dirname(__file__)  # NOQA: E402
    data_folder = os.path.join(dirname, '../../../../../data/processed/onto5/flair/uncased')

    corpus = NLPTaskDataFetcher.load_column_corpus(
        data_folder,
        columns,
        train_file='train.iob2',
        test_file='test.iob2',
        dev_file='dev.iob2'
    )

    print(corpus.train[0].to_tagged_string('ner'))

if __name__ == '__main__':
    train_model()