# Faster Repeated Evasion Attacks in Tree Ensembles

This repository is the official implementation of the [paper](https://openreview.net/pdf?id=Ugr0yPzY71) "Faster Repeated Evasion Attacks in Tree Ensembles", presented at NeurIPS 2024. 

## Requirements

Activate your environment, e.g., using `venv`:
```
python -m venv my_new_venv
source my_new_venv/bin/activate
```

Install dependencies:
```
pip install click dtai-veritas groot-trees gurobipy numpy pandas peewee scikit-learn scipy tqdm xgboost
pip install git+https://github.com/laudv/prada.git
```

## Reproducing Experiments

To reproduce experiments from the paper, run this command:

```
python main.py [-m model_type] [-a attack_type] dataset
```

## Cite this work

Cascioli L., Devos L., Kuzelka O., Davis J., "Faster Repeated Evasion Attacks in Tree Ensembles", NeurIPS 2024.
