import numpy as np
from noisy_classification_template import corrupt_training_labels


class FakeRng:
    """Return predetermined random values for a deterministic test."""

    def __init__(self, values):
        self.values = np.array(values)

    def random(self, shape):
        assert shape == self.values.shape
        return self.values.copy()


clean_labels = np.array([1, 1, -1, -1, 1, -1])
clean_copy = clean_labels.copy()

# Values are compared against:
# +1 labels: 0.35
# -1 labels: 0.20
rng = FakeRng(
    [
        0.34,  # +1: flips because 0.34 < 0.35
        0.35,  # +1: does not flip because 0.35 is not < 0.35
        0.19,  # -1: flips because 0.19 < 0.20
        0.20,  # -1: does not flip because 0.20 is not < 0.20
        0.36,  # +1: does not flip
        0.21,  # -1: does not flip
    ]
)

actual = corrupt_training_labels(clean_labels, rng)

expected = np.array([-1, 1, 1, -1, 1, -1])

assert np.array_equal(actual, expected)
assert actual.shape == clean_labels.shape
assert np.issubdtype(actual.dtype, np.integer)

# Verify that the original labels were not modified.
assert np.array_equal(clean_labels, clean_copy)

# Verify that labels remain valid.
assert np.all(np.isin(actual, [-1, 1]))

print("corrupt_training_labels test passed!")
