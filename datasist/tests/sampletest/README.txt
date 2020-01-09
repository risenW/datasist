
    PROJECT STRUCTURE:

            ├── data
            │   ├── processed
            │   └── raw
            ├── outputs
            │   ├── models
            ├── src
            │   ├── scripts
            │       ├── ingest
            │       ├── modeling
            │       ├── preparation
            │       ├── test
            ├   ├── notebooks


            data: Stores data used for the experiments, including raw and intermediate processed data.
                processed: stores all processed data files after cleaning, analysis, feature creation etc.
                raw: Stores all raw data obtained from databases, file storages, etc.

            outputs:Stores all output files from an experiment.
                models: Stores trained binary model files. This are models saved after training and evaluation for later use.

            src: Stores all source code including scripts and notebook experiments.
                scripts: Stores all code scripts usually in Python/R format. This is usually refactored from the notebooks.
                    modeling: Stores all scripts and code relating to model building, evaluation and saving.
                    preparation: Stores all scripts used for data preparation and cleaning.
                    ingest: Stores all scripts used for reading in data from different sources like databases, web or file storage.
                    test: Stores all test files for code in scripts.
                notebooks: Stores all jupyter notebooks used for experimentation.

    