import sys
import os
import json

dirname = os.path.dirname(__file__)  # NOQA: E402
filename = os.path.join(dirname, '../')  # NOQA: E402
sys.path.append(filename)  # NOQA: E402

from data.make_dataset_onto import get_dataset
from baseline_model import get_statistics, entities_equal

documents = get_dataset()

statistics = get_statistics(documents)

print("statistics: ")
print(json.dumps(statistics, indent=4))
