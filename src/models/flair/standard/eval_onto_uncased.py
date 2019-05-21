import sys
import os
import json
from rasa_nlu.model import Interpreter

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../'))  # NOQA: E402
sys.path.append(os.path.join(dirname, '../../../'))  # NOQA: E402

from data.make_dataset_onto import get_dataset
from baseline_model import get_statistics

def evaluate():
    documents = get_dataset(cased=False)

    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, '../../../../models/flair_standard/default/current')
    interpreter = Interpreter.load(model_path)

    statistics = get_statistics(documents, interpreter)

    print("statistics: ")
    print(json.dumps(statistics, indent=4))

    f = open("results/onto_uncased.json", "w+")
    f.write(json.dumps(statistics, indent=4))
    f.close()

if __name__ == "__main__":
    evaluate()