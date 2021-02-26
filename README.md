# Salaries-Project

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Salaries prediction project in Python.

## Directory structure

├── README.md
├── code
│   ├── Salaries\ prediction.ipynb
│   ├── eda
│   │   ├── plot.py
│   │   └── stats.py
│   ├── exceptions.py
│   ├── feature_engineering
│   │   └── encoders.py
│   ├── model_selection.py
│   └── preprocessing.py
├── cross_val_logs
│   ├── engineered_train_data_cross_val.csv
│   └── train_data_cross_val.csv
├── data
│   ├── test_features.csv
│   ├── train_features.csv
│   └── train_salaries.csv
└── results
    └── 2021_02_25_03_27_24
        ├── feature_importances.csv
        ├── model.joblib
        └── predictions.csv