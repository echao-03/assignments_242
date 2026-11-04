import numpy as np
from kmeans_template import update_centroids


data = np.array([
    [0.0, 0.0],
    [2.0, 2.0],
    [10.0, 10.0],
    [12.0, 12.0],
])

old_centroids = np.array([
    [0.0, 0.0],
    [10.0, 10.0],
])

assignments = np.array([0, 0, 1, 1])
old_copy = old_centroids.copy()

actual = update_centroids(data, old_centroids, assignments)
expected = np.array([
    [1.0, 1.0],
    [11.0, 11.0],
])

assert actual.shape == (2, 2)
assert np.allclose(actual, expected)
assert np.issubdtype(actual.dtype, np.floating)
assert np.array_equal(old_centroids, old_copy)

print("Non-empty centroid test passed!")


# Cluster 2 has no assigned points. The expected policy is to retain
# its previous centroid.
empty_data = np.array([
    [0.0, 0.0],
    [2.0, 2.0],
    [10.0, 10.0],
])

empty_centroids = np.array([
    [0.0, 0.0],
    [10.0, 10.0],
    [20.0, 20.0],
])

empty_assignments = np.array([0, 0, 1])

actual = update_centroids(
    empty_data,
    empty_centroids,
    empty_assignments,
)

expected = np.array([
    [1.0, 1.0],
    [10.0, 10.0],
    [20.0, 20.0],
])

assert np.allclose(actual, expected)

print("Empty cluster test passed!")
print("All centroid tests passed!")