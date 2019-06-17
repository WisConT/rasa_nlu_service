import os
import warnings

# warnings.simplefilter('ignore', ruamel.yaml.error.UnsafeLoaderWarning)

from rasa_core.agent import Agent
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.interpreter import RasaNLUInterpreter

dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, '../../../../../models/rasa/flair/standard/default/current')
interpreter = RasaNLUInterpreter(model_path)
# interpreter = NaturalLanguageInterpreter.create(model_directory)
dialogue_path = os.path.join(dirname, '../../../../../models/rasa/flair/final/dialogue')
agent = Agent.load(dialogue_path, interpreter=interpreter)

print("Your bot is ready to talk! Type your messages here or send 'stop'")

while True:
    a = input()

    if a == 'stop':
        break

    responses = agent.handle_text(a)

    print(responses)

    for response in responses:
        print(response["text"])