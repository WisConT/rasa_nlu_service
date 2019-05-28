from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import warnings
import os

from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy


def train_nlu():
    dirname = os.path.dirname(__file__)

    training_data = load_data(os.path.join(dirname, 'resources/nlu-data.md'))
    trainer = Trainer(config.load(
        os.path.join(dirname, 'resources/nlu-config.yml')))
    trainer.train(training_data)

    model_path = os.path.join(dirname, '../../../../models/rasa/spacy/onto_cased/')
    model_directory = trainer.persist(model_path, fixed_model_name='current')

    return model_directory


def train_dialogue(domain_file=None, model_path=None, training_data_file=None):
    dirname = os.path.dirname(__file__)
    domain_file = os.path.join(
        dirname, 'resources/domain.yml') if domain_file == None else domain_file
    model_path = os.path.join(
        dirname, '../../../../models/dialogue/spacy_onto_cased/') if model_path == None else model_path
    training_data_file = os.path.join(
        dirname, 'resources/stories.md') if training_data_file == None else training_data_file

    agent = Agent(
        domain_file,
        policies=[MemoizationPolicy(max_history=3), KerasPolicy()]
    )
    training_data = agent.load_data(training_data_file)
    agent.train(training_data)

    agent.persist(model_path)

    return agent


def train_all():
    model_directory = train_nlu()
    agent = train_dialogue()
    return [model_directory, agent]


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    utils.configure_colored_logging(loglevel='INFO')

    parser = argparse.ArgumentParser(description='starts the bot training')
    parser.add_argument(
        'task',
        choices=['train-nlu', 'train-dialogue', 'train-all'],
        help='what should the bot do?'
    )
    task = parser.parse_args().task

    if task == "train-nlu":
        train_nlu()
    elif task == "train-dialogue":
        train_dialogue()
    elif task == "train-all":
        train_all()
