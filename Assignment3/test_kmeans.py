import numpy as np
from kmeans_template import update_assignments


data = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [9.0, 9.0],
    [10.0, 10.0],
])

centroids = np.array([
    [0.0, 0.0],    # cluster 0
    [10.0, 10.0],  # cluster 1
])

expected = np.array([0, 0, 1, 1])
actual = update_assignments(data, centroids)

print("Actual:  ", actual)
print("Expected:", expected)

assert actual.shape == (4,)
assert np.issubdtype(actual.dtype, np.integer)
assert np.array_equal(actual, expected)

tie_data = np.array([[5.0, 0.0]])
tie_centroids = np.array([
    [0.0, 0.0],
    [10.0, 0.0],
])

tie_result = update_assignments(tie_data, tie_centroids)

assert np.array_equal(tie_result, np.array([0]))
print("Tie test passed!")

print("All tests passed!")