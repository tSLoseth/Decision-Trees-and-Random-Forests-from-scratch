# Decision Trees and Random Forests from Scratch

> Completed for **INF264 — Introduction to Machine Learning**, University of Bergen, fall 2025.
> Final grade: **100%**.

**Authors:** Thomas Strønstad-Løseth · Trym Oscar Lillevik Bårdsen

A from-scratch implementation in NumPy of a greedy **ID3 decision tree** and a
**random forest**, benchmarked against scikit-learn's tuned classifiers on a
subset of the Letter Recognition dataset.

## Files

- [`decision_tree.py`](decision_tree.py) — ID3 decision tree classifier
- [`random_forest.py`](random_forest.py) — random forest classifier built on the decision tree
- [`run_experiments.ipynb`](run_experiments.ipynb) — training, hyperparameter tuning, cross-validation, and comparison with scikit-learn

## Report

The full project report is included as
[`INF264_project1_report.pdf`](INF264_project1_report.pdf).

## Required packages

```
pip install numpy pandas matplotlib scikit-learn
```

## Note on data

The `letters.csv` dataset (a subset of the UCI Letter Recognition dataset) is
**not included** in this repository. Place it in the same directory as the
notebook before running.

## How to run

With `letters.csv` in place, run all cells in `run_experiments.ipynb` to
reproduce the results from the report.
