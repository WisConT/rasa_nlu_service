import tweepy
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy.error import TweepError
import json
import pandas as pd
import os
from pathlib import Path

dirname = os.path.dirname(__file__)
data_external_dir = os.path.join(dirname, '../../data/external')
data_interim_dir = os.path.join(dirname, '../../data/interim')
# data_processed_dir = os.path.join(dirname, '../../data/processed')


def get_twitter_api():
    consumer_key = 'mNjU3LqiSVc6yGTw9JwFOFbzy'
    consumer_secret = 'QcYAJpDDn0p2vXCN7hISvnMXn39XpQ6EhTXG1tmB9mnqvUlo7D'
    access_token = '228960355-SgRxRURKJRN4UNhEyrL0gW6q8zdFBLEw3NOCP5dq'
    access_secret = 'cp8NC6DtuUyaKnfRQyRZoyWbdOPQU7ZXVuLuJgdKPJ06A'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    return api


def get_raw_tweet_from_id(api, string):
    return api.get_status(string)


def combine_tsv(outname, microposts_dir):
    extension = 'gold.tsv'
    all_filenames = [Path(data_external_dir)/microposts_dir/f for f in os.listdir(
        Path(data_external_dir)/microposts_dir) if (extension in f)]
    combined_csv = pd.concat([pd.read_csv(f, sep='\t', header=None, names=[
                             "id", "start", "end", "entry", "entity"]) for f in all_filenames])
    combined_csv.to_csv(Path(data_interim_dir)/"microposts2015" /
                        outname, index=False, encoding='utf-8-sig', sep='\t')
    return Path(data_interim_dir)/"microposts2015"/outname


def main():
    microposts_dir = "microposts2015-neel_challenge_gs"
    outname = microposts_dir + "_all.csv"
    fullname = combine_tsv(outname, microposts_dir)
    examples = {}
    api = get_twitter_api()

    with open(fullname) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            split = line.split('\t')
            if idx % 1000 == 0:
                print("Line", idx)
            try:
                tweet = get_raw_tweet_from_id(api, split[0])
            except TweepError:
                continue
            text = tweet.text
            if text not in examples:
                examples[text] = {"entities": [],
                                  "intent": "greeting", "text": text}
            start = int(split[1])
            end = int(split[2])

            examples[text]["entities"].append(
                {"entity": split[-1], "start": start, "end": end, "value": text[start:end]})

    nlu_data = {"rasa_nlu_data": {
        "common_examples": list(examples.values()),
        "regex_features": [],
        "lookup_tables": [],
        "entity_synonyms": []
    }}

    with open(dirname + '/../../data/processed/microposts2015/neel-nlu-data.json', 'w') as outfile:
        json.dump(nlu_data, outfile, separators=(',', ':'), indent=4)


if __name__ == "__main__":
    main()
