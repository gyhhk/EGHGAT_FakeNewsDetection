Make sure the following files are present as per the directory structure before running the code,

```
GHGAT_FakeNewsDetection
├── README.md
├── *.py
└───models
|   └── *.py 
└───data
    ├── fakeNews
    │   ├── adjs
    │   │   ├── train
    │   │   ├── dev
    │   │   └── test
    │   ├── fulltrain.csv
    │   ├── balancedtest.csv
    │   ├── test.xlsx
    │   ├── entityDescCorpus.pkl
    │   └── entity_feature_transE.pkl
    └── stopwords_en.txt

```

# Dependencies

Our code runs on the Tesla P100-PCIE-16GB GPU, with the following packages installed:

```
python 3.7.12
torch 1.11.0
nltk 3.6.5
tqdm
numpy
pandas
matplotlib
scikit_learn
xlrd (pip install xlrd)
```

# Run

Train and test,

```
python main.py --mode 0
```

Test,

```
python main.py --mode 1 --model_file MODELNAME
```

