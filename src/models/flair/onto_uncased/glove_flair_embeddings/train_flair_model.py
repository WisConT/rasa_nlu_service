from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
import os

def train_model():
    columns = {0: 'text', 1: 'ner'}

    dirname = os.path.dirname(__file__)  # NOQA: E402
    data_folder = os.path.join(dirname, '../../../../../data/processed/onto5/uncased')

    corpus = NLPTaskDataFetcher.load_column_corpus(
        data_folder,
        columns,
        train_file='train.conll',
        test_file='test.conll',
        dev_file='dev.conll'
    )

    print(corpus)
    return

    tag_type = 'ner'

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    embedding_types = [
        WordEmbeddings('crawl'),
        FlairEmbeddings('news-forward', use_cache=True),
        FlairEmbeddings('news-backward', use_cache=True)
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
    )

    trainer = ModelTrainer(tagger, corpus)

    model_location = os.path.join(dirname, '../../../../../models/flair/onto_uncased/glove_flair_embeddings')

    trainer.train(
        model_location,
        learning_rate=0.1,
        embeddings_in_memory=False
    )

if __name__ == '__main__':
    train_model()