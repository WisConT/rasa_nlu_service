import sys
import os

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

from data.make_dataset_onto import get_dataset
from baseline_model import get_statistics, entities_equal

documents = get_dataset()

(recall, precision, f1) = get_statistics(documents)

print("recall: " + str(recall))
print("precision: " + str(precision))
print("f1: " + str(f1))
