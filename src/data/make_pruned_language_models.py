import spacy
from spacy.language import Language
from pathlib import Path
import os

dirname = os.path.dirname(__file__)
language_dir = os.path.join(dirname, '../../models/language/')

nlp_lg = spacy.load('en_core_web_md')
nlp = nlp_lg

print("Pruning...")
n_vectors = 417194
removed_words = nlp.vocab.prune_vectors(n_vectors, 2048)

nlp.vocab.vectors.resize((417194, 300))

print("Number of keys:", len(nlp.vocab.vectors.keys()))
print("Number of values:", len(nlp.vocab.vectors))

nlp.to_disk(Path(language_dir)/"pruned_md")
