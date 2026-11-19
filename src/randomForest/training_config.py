"""Editable Random Forest settings used by train_random_forest.py."""

# Forest size and tree complexity
N_ESTIMATORS = 1000
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"
CRITERION = "gini"

# Class balance and positive-class decision threshold
CLASS_WEIGHT = None
DECISION_THRESHOLD = 0.5  # Lower it to favor phishing recall over precision.

# Reproducibility and terminal progress
RANDOM_STATE = 42
PROGRESS_EVERY = 50