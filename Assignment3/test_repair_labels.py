import numpy as np
from noisy_classification_template import repair_labels_knn


# Two well-separated groups. Each group has six points.
train_features = np.array([
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [0.1, 0.1],
    [0.2, 0.0],
    [0.0, 0.2],

    [10.0, 10.0],
    [10.1, 10.0],
    [10.0, 10.1],
    [10.1, 10.1],
    [10.2, 10.0],
    [10.0, 10.2],
])

# Each group has one intentionally incorrect label.
noisy_labels = np.array([
     1,  1,  1,  1, -1,  1,
    -1, -1, -1, -1,  1, -1,
])

original_labels = noisy_labels.copy()

repaired_labels = repair_labels_knn(
    train_features,
    noisy_labels,
    neighbors=5,
)

# The majority label in the first group is +1.
# The majority label in the second group is -1.
expected_labels = np.array([
     1,  1,  1,  1,  1,  1,
    -1, -1, -1, -1, -1, -1,
])

assert repaired_labels.shape == noisy_labels.shape
assert np.issubdtype(repaired_labels.dtype, np.integer)
assert np.array_equal(repaired_labels, expected_labels)

# The function must not modify the original noisy labels.
assert np.array_equal(noisy_labels, original_labels)

# Only valid binary labels should be returned.
assert np.all(np.isin(repaired_labels, [-1, 1]))

print("repair_labels_knn test passed!")