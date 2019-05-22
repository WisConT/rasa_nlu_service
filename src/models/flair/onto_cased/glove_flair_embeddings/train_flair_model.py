from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
import os

def train_model():
    columns = {0: 'text', 1: 'ner'}

    dirname = os.path.dirname(__file__)  # NOQA: E402
    data_folder = os.path.join(dirname, '../../../../../data/processed/onto5/flair/cased')

    corpus = NLPTaskDataFetcher.load_column_corpus(
        data_folder,
        columns,
        train_file='train.iob2',
        test_file='test.iob2',
        dev_file='dev.iob2'
    )

    tag_type = 'ner'

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward')
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True
    )

    trainer = ModelTrainer(tagger, corpus)

    model_location = os.path.join(dirname, '../../../../../models/flair/onto_cased/glove_flair_embeddings')

    trainer.train(
        model_location,
        max_epochs=150
    )

if __name__ == '__main__':
    train_model()