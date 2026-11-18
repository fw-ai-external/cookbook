"""User-actionable training configuration and data errors."""


class UserConfigError(Exception):
    """Base class for user-actionable configuration errors raised inside a recipe."""


class DatasetError(UserConfigError):
    """Raised when a dataset is malformed or yields no trainable examples.

    Typical causes: invalid JSONL/preference-row shape, every row filtered by
    sequence length, or no assistant tokens selected by ``train_on_what`` /
    per-message ``weight``.
    """


NO_VALID_TRAINING_EXAMPLES_MESSAGE = (
    "No valid training examples remained after tokenization. Verify that the "
    "dataset contains trainable assistant messages for the selected "
    "train_on_what setting and that examples fit the configured maximum "
    "sequence length, then retry."
)

NO_VALID_PREFERENCE_PAIRS_MESSAGE = (
    "No valid preference pairs remained after tokenization. Verify that the "
    "dataset contains valid chosen and rejected examples that fit the "
    "configured maximum sequence length, then retry."
)


class WandbConfigError(UserConfigError):
    """Raised when Weights & Biases auth/config is invalid (bad key, entity, or project)."""
