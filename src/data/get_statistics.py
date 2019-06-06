# concept net query
import plac
from rasa_nlu.training_data import load_data
import urllib
from urllib.error import HTTPError
import urllib3
import json
import os
from pathlib import Path
import time

dirname = os.path.dirname(__file__)
stats_per_sf_dir = os.path.join(
    dirname, '../../data/processed/surface_form_stats')


class Result:
    def __init__(self, json_data):
        result = {}
        for key, value in json_data.items():
            result[key] = value
        self.result = result


class Search:
    def __init__(self, **kwargs):
        query_args = {}
        # print(kwargs)
        for key, value in kwargs.items():
            query_args[key] = value
        self.encoded_query_args = urllib.parse.urlencode(query_args)

    def search(self):
        url = ''.join(['%s%s' % ("http://api.conceptnet.io/search", '?')]
                      ) + self.encoded_query_args
        try:
            json_data = make_http_request(url)
            time.sleep(0.5)
        except HTTPError as e:
            print("HTTPERROR", e)
            exit()
        return json_data


def make_http_request(url):
    request = urllib.request.Request(url)
    data = urllib.request.urlopen(request)
    return json.load(data)


@plac.annotations(
    test_data_path=("test data", "positional", None, str)
)
def main(test_data_path=None):
    nlu_data = load_data(test_data_path)
    sorted_entities = nlu_data.sorted_entities()
    entity_values = set([ent["value"].lower() for ent in sorted_entities])
    print("No. of entities", len(entity_values))
    sf_counts = []
    for idx, val in enumerate(entity_values):
        if idx % 100 == 0:
            print(idx)
        search = Search(node="/c/en/" + val.replace(" ", "_"), rel="/r/IsA")
        data = search.search()
        count = len(data["edges"])
        if count == 0:
            count = 1
        sf_counts.append(count)

    dict_sf_counts = dict(zip(entity_values, sf_counts))

    with open(Path(stats_per_sf_dir)/((test_data_path.split("/")[-1].split("-")[0]) + ".json"), "w") as outfile:
        json.dump(dict_sf_counts, outfile, separators=(',', ':'), indent=4)


if __name__ == "__main__":
    plac.call(main)
