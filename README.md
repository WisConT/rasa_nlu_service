Github: https://github.com/WisConT/rasa_nlu_service/tree/jason

Some of the code in the src/data and src/models directory was written by Tim Green, the other contributor to the rasa_nlu_service GitHub. These include all files of the form "src/data/make_dataset_*" and "src/models/baseline_*" as they were used for his evaluation of the datasets. I made use of some of methods in his code to extract documents from the external datasets.

Inspirations for code are noted where indicated.

The spaCy directory will appear empty as it was where the spaCy library was forked as a submodule. A patchfile is included which can be applied to the spaCy library found below to see the changes that were made.
SpaCy Github: https://github.com/explosion/spaCy

The models/pipeline directory is also empty as trained pipelines can take up to GBs of storage.