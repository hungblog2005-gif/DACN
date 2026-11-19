"""Training and model hyperparameters for train_phobert.py.

Edit these values before training. Command-line arguments in train_phobert.py
can override the matching settings.
"""

# Pretrained Vietnamese language model
MODEL_NAME = "vinai/phobert-base"

# Input/tokenization
MAX_LENGTH = 128

# Training loop
EPOCHS = 3.0
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "linear"
MAX_GRAD_NORM = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
LABEL_SMOOTHING_FACTOR = 0.0

# PhoBERT classification-head/model dropout
CLASSIFIER_DROPOUT = 0.1
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROB = 0.1
GRADIENT_CHECKPOINTING = False

# Runtime and reproducibility
LOGGING_STEPS = 100
SEED = 42
FP16 = False  # Set True for a compatible CUDA GPU.
SAVE_TOTAL_LIMIT = 2

