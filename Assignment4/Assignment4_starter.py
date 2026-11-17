"""Optional starter code for CSE 242 Assignment 4.

You do not need to use this file. Add, remove, or reorganize functions as useful.
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


def update_assignments(X, centroids):
    """Return the index of the closest centroid for each observation."""
    retValue = []

    for point in X:
        curCentroid = float("inf")
        centroidIndex = None
        for i in range(len(centroids)):
            distance = np.linalg.norm(point - centroids[i])
            if distance < curCentroid:
                curCentroid = distance
                centroidIndex = i

        retValue.append(centroidIndex)

    return np.array(retValue, dtype=int)


def update_centroids(X, centroids, assignments):
    """Return updated centroids for k clusters."""
    new_centroids = centroids.astype(float)
    for i in range(len(centroids)):
        # Obtain the cluster point coordinates that are connected to the current centroid
        cluster_points = X[assignments == i]
        # print("cluster_point:", i,  cluster_points)
        # If the cluster is not empty, obtain the mean of the XY coordinates that is associated with the centroid and apply it to the aray
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)

    return new_centroids


def kmeans(X, centroids, max_iterations=100, tol=1e-6, random_state=None):
    """Fit K-means from scratch and return centroids, assignments, and history."""
    for _ in range(max_iterations):
        assignments = update_assignments(X, centroids)
        centroids = update_centroids(X, centroids, assignments)
    assignments = update_assignments(X, centroids)
    
    return centroids, assignments

# Obtained from Assignment 3
def get_initial_clusters(k, data, rng):
    """Sample k rows with replacement, as in the source's randint initializer.

    This adaptation accepts a seeded NumPy generator for reproducibility.
    Repeated initial rows are possible; your empty-cluster policy must cope.
    """
    return data[rng.integers(0, data.shape[0], size=k)].copy()


def corrupt_labels(y, noise_config, random_state=None):
    """Create a reproducible noisy version of y according to your design."""
    raise NotImplementedError


def repair_labels(X, y_noisy, **kwargs):
    """Optional structure for your nearest-neighbor or clustering repair method."""
    raise NotImplementedError

K_VALUES = [4, 6, 8, 10]
def main(seed = 0, max_iterations = 100):
    # Iterate through .csv data, grabbing only countries with 2014 data
        # Life expectancy, Adult mortality, Alcohol, and GDP
    newData = {}
        
    with open('Life_Expectancy_Data.csv') as fileObj:
        # Used csv.DictReader for easier readability when iterating
        readerObj = csv.DictReader(fileObj)
        for curRow in readerObj:
            if curRow["Year"] == "2014":
                country = curRow["Country"]
                year = int(curRow["Year"])
                life_expectancy = float(curRow["Life expectancy"])
                adult_mortality = float(curRow["Adult Mortality"])
                alcohol = float(curRow["Alcohol"])
                gdp = float(curRow["GDP"])
                newData[country] = {"Year": year, "life_expectancy": life_expectancy, "adult_mortality": adult_mortality, "Alcohol": alcohol, "GDP": gdp}
                
        
    X = np.array([
    [
        row["life_expectancy"],
        row["adult_mortality"],
        row["Alcohol"],
        row["GDP"]
    ]
    for row in newData.values()
    ], dtype=float)
    
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)

    X_scaled = (X - feature_means) / feature_stds
    rng = np.random.default_rng(seed)
    for current_k in K_VALUES:
        initial_centroids = get_initial_clusters(current_k, X_scaled, rng)
        
        current_centroids, current_assignments = kmeans(X_scaled, initial_centroids, max_iterations)
        print("centroids:", current_centroids, "assignments:", current_assignments)
    
    # Use PDA to reduce dimentionality
    # Determine which k value is needed for clustering
    



if __name__ == "__main__":
    main()
    
