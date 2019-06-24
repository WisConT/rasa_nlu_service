import xml.etree.ElementTree as ET
import os
import json
from pathlib import Path

dirname = os.path.dirname(__file__)
newsreader_dir = os.path.join(
    dirname, '../../data/external/meantime_newsreader_english_oct15/intra-doc_annotation/')
stats_per_sf_dir = os.path.join(
    dirname, '../../data/external/stats_per_surface_form')


def extract_entities(ref_map, ent_mentions, ent_instances, sentence, xml):
    ents = []
    sent_tids = [tok.attrib["t_id"] for tok in sentence]
    text = " ".join([tok.text for tok in sentence])

    lens = [len(tok.text) for tok in sentence]

    for em in ent_mentions:
        anchors = em.findall("token_anchor")
        tids = [an.attrib["t_id"] for an in anchors]
        start_index = sent_tids.index(tids[0])
        end_index = sent_tids.index(tids[-1])
        start = sum(lens[0:start_index]) + start_index
        end = sum(lens[start_index:end_index + 1]) + \
            end_index - start_index + start

        mid = em.attrib["m_id"]
        try:
            ent_id = ref_map[str(mid)]
        except KeyError as error:
            print("Key error:", error)
            print(xml)
            exit()

        ent = {
            "entity": [inst.attrib["ent_type"] for inst in ent_instances if inst.attrib["m_id"] == ent_id][0],
            "value": " ".join([tok.text for tok in sentence if tok.attrib["t_id"] in tids]),
            "start": start,
            "end": end
        }

        ents.append(ent)
    return ents, text


def extract_sentences(all_tokens):
    sents = []
    curr_sent = 0
    for tok in all_tokens:
        if int(tok.attrib["sentence"]) != curr_sent:
            curr_sent = curr_sent + 1
        if len(sents) <= curr_sent:
            sents.append([])
        sents[curr_sent].append(tok)
    return sents


def parse_newsreader_xml(xml, examples):
    tree = ET.parse(xml)
    root = tree.getroot()

    all_tokens = root.findall("token")
    all_sentences = extract_sentences(all_tokens)
    markables = root.findall("Markables")[0]
    all_ent_mentions = []
    for em in markables.findall("ENTITY_MENTION"):
        if ("syntactic_type" in em.attrib) \
            and (em.attrib["syntactic_type"] == "NAM"
                 or em.attrib["syntactic_type"] == "PRE.NAM"):
            all_ent_mentions.append(em)

    ent_instances = markables.findall("ENTITY")

    relations = root.findall("Relations")[0]
    ref_map = {str(s.attrib["m_id"]): ref.findall("target")[0].attrib["m_id"]
               for ref in relations.findall("REFERS_TO")
               for s in ref.findall("source")}
    # print ([[s.attrib["m_id"] for s in ref.findall("source")] for ref in relations.findall("REFERS_TO")])

    for sent in all_sentences:
        tids = [tok.attrib["t_id"] for tok in sent]
        ent_mentions = []
        for em in all_ent_mentions:
            for an in em:
                if an.attrib["t_id"] in tids:
                    ent_mentions.append(em)
                    continue
        ents, text = extract_entities(ref_map, set(
            ent_mentions), ent_instances, sent, xml)
        example = {"intent": "greeting", "entities": ents,
                   "text": text, "xml": str(xml)}
        if len(example["entities"]) != 0:
            examples.append(example)


def parse_dir(dir_name, examples):
    for xml in os.listdir(dir_name):
        if ".DS_Store" in xml:
            continue
        parse_newsreader_xml(Path(dir_name)/xml, examples)


def main():
    examples = []
    for d in os.listdir(newsreader_dir):
        if "corpus" in d:
            parse_dir(Path(newsreader_dir)/d, examples)
    print("No. of examples:", len(examples))
    path_to_sf_data = Path(stats_per_sf_dir) / \
        "wikinews.json"
    with open(path_to_sf_data) as data_file:
        sf_data = json.load(data_file)
    get_sf_stats(sf_data, examples)
    nlu_data = {"rasa_nlu_data": {
        "common_examples": examples,
        "regex_features": [],
        "lookup_tables": [],
        "entity_synonyms": []
    }}
    with open(dirname + '/../../data/processed/newsreader/newsreader-nlu-data.json', 'w') as outfile:
        json.dump(nlu_data, outfile, separators=(',', ':'), indent=4)


if __name__ == "__main__":
    main()
