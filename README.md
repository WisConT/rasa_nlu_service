# Chatbot Project

## Data Organisation

### OntoNotes 5.0

Steps to import data:
1. After downloading the OntoNotes 5.0 data set, copy the folder `data/files/data` from the data set and paste it in the `data/external` folder in this project.
2. Rename the directory `onto5`.
3. Run `python3 src/data/clean_dataset_onto.py` to convert the data to the desired format. This will create a formatted copy of the data in `data/interim`.
4. You will now be able to run `python3 src/models/baseline_onto_model.py` with the formatted data.

Example of formatted data:

```
With	O
their	O
unique	O
charm	O
,	O
these	O
well	O
-	O
known	O
cartoon	O
images	O
once	O
again	O
caused	O
Hong	B-GPE
Kong	I-GPE
to	O
be	O
a	O
focus	O
of	O
worldwide	O
attention	O
.	O
```

### CoNLL 2003

The WNUT data should be a tsv of four columns, the first is the tokens and the fourth is the entity classification. It should be saved in a file in this location `data/interim/conll_2003/test.txt`. Example format:

```
Japan NNP B-NP B-LOC
began VBD B-VP O
the DT B-NP O
defence NN I-NP O
of IN B-PP O
their PRP$ B-NP O
Asian JJ I-NP B-MISC
Cup NNP I-NP I-MISC
title NN I-NP O
with IN B-PP O
a DT B-NP O
lucky JJ I-NP O
2-1 CD I-NP O
win VBP B-VP O
against IN B-PP O
Syria NNP B-NP B-LOC
in IN B-PP O
a DT B-NP O
Group NNP I-NP O
C NNP I-NP O
championship NN I-NP O
match NN I-NP O
on IN B-PP O
Friday NNP B-NP O
. . O O
```

### WNUT2017

The WNUT data should be a tsv of two columns, the first is the tokens and the second is the entity classification. It should be saved in a file in this location `data/interim/wnut_2017/test.txt`. Example format:

```
The	O
soldier	O
was	O
killed	O
when	O
another	O
avalanche	O
hit	O
an	O
army	O
barracks	O
in	O
the	O
northern	O
area	O
of	O
Sonmarg	B-location
,	O
said	O
a	O
military	O
spokesman	O
.	O
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
