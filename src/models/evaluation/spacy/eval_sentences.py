import os
import time
import spacy

def eval_input(model, user_input):
    start_timestamp = time.time()
    result = model(user_input)
    end_timestamp = time.time()

    print("time taken: " + str(end_timestamp - start_timestamp))

    for entity in result.ents:
        print(entity.label_ + ': ' + entity.text)

    print()

def start():
    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, '../../../../models/spacy/onto_uncased/model-best')
    model = spacy.load(model_path)

    print("Type your sentence...")
    user_input = input()

    while user_input != 'exit':
        eval_input(model, user_input)
        user_input = input()


if __name__ == '__main__':
    start()