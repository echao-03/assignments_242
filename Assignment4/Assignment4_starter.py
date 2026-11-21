"""Optional starter code for CSE 242 Assignment 4.

You do not need to use this file. Add, remove, or reorganize functions as useful.
"""
import argparse
from pathlib import Path
from collections import defaultdict


import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from matplotlib import colormaps
import pandas as pd

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors





def update_assignments(X, centroids):
    """Return the index of the closest centroid for each observation."""
    retValue = []

    for point in X:
        curCentroid = float("inf")
        centroidIndex = None
        for i in range(len(centroids)):
            distance = np.linalg.norm(point - centroids[i])
            
            # To deal with ties, the current point that has a curCentroid assigned will be prioritized
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

# Obtained from Assignment 3
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
        

""" Created a asymmetrical design for corrupting the labels, if the current rng is less than 
    0.10, change label to "Developing". If it's below 0.30, but above 0.10, then change the label to "Developed
    This makes the probability: Developed: 20%, Developing: 10%, keep label: 70%"""
    
"""If the noise_config is symmetrical, then it follows: If the rng is less than 0.30, change label to "Developed". 
   Else, change the label to "Developing. """
def corrupt_labels(y, noise_config, seed = 0):
    """Create a reproducible noisy version of y according to your design."""
    corrupt_label = np.asarray(y, dtype = object).copy()
    
    rng = np.random.default_rng(seed)
    if noise_config == "Asymmetric":
        for idx, label in enumerate(corrupt_label):
            if label == "Developing":
                if rng.random() < 0.30:
                    corrupt_label[idx] = "Developed"
            
            elif label == "Developed":
                if rng.random() < 0.10:
                    corrupt_label[idx] = "Developing"
                    
            else:
                # Edge case to check if label is not expected
                return ValueError(f"Unexpected label: {label}")
         
    # If noise_config is "symmetric"        
    else:
        for idx, label in enumerate(corrupt_label):
            if rng.random() < 0.30:
                if label == "Developing":
                    corrupt_label[idx] = "Developed"
                else:
                    corrupt_label[idx] = "Developing"
                    
    return corrupt_label
    
    

"""Repairing labels based on KNN. The neighbors are fitted """
def repair_labels(X, y_noisy, neighbors):
    """Optional structure for your nearest-neighbor or clustering repair method."""
    
    near_neighbors = NearestNeighbors(n_neighbors = neighbors + 1, algorithm = "auto").fit(X)
    
    near_neighbors.fit(X)
    
    _, indices = near_neighbors.kneighbors(
        X
    ) 
    neighbor_indices = indices[:, 1:]
    
    neighbor_labels = y_noisy[neighbor_indices]
    
    repaired_labels = y_noisy.copy()
    
    developed_counts = np.sum(
        neighbor_labels == "Developed",
        axis=1,
    )
    
    majority = neighbors // 2 + 1
    
    repaired_labels[developed_counts >= majority] = "Developed"
    repaired_labels[developed_counts <= neighbors // 2] = "Developing"
    
    return repaired_labels

    

def within_cluster_wcss(X, centroids, assignments):
    wcss = 0.0

    for cluster_id in range(len(centroids)):
        cluster_points = X[assignments == cluster_id]

        if len(cluster_points) > 0:
            differences = cluster_points - centroids[cluster_id]
            wcss += np.sum(differences ** 2)

    return wcss

K_VALUES = [4, 6, 8, 10, 12, 14, 16,  18, 20, 22, 24, 26, 28, 30]

def kmeans_experiments(output, seed = 0, max_iterations = 100):
        # Iterate through .csv data, grabbing only countries with 2014 data
        # Life expectancy, Adult mortality, Alcohol, and GDP
        
    output.mkdir(parents=True, exist_ok=True)
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
    
    wcss_values = []
    
    for current_k in K_VALUES:
        initial_centroids = get_initial_clusters(current_k, X_scaled, rng)
        
        current_centroids, current_assignments = kmeans(X_scaled, initial_centroids, max_iterations)
        
        
        pca = PCA(n_components = 2)
        reduced_PCA = pca.fit_transform(X_scaled)
        reduced_centroids = pca.transform(current_centroids)
        
        score = silhouette_score(X_scaled, current_assignments)
        
        # wcss = within_cluster_wcss(
        #         X_scaled,
        #         current_centroids,
        #         current_assignments
        #         )
        #         wcss_values.append(wcss)
        
        # print(f"k = {current_k}, {score}") 
        
        
        cluster_cmap = colormaps["viridis"].resampled(current_k)
        legend_handles = []

        for cluster_id in range(current_k):
            in_cluster = current_assignments == cluster_id

            handle = plt.scatter(
                reduced_PCA[in_cluster, 0],
                reduced_PCA[in_cluster, 1],
                c=[cluster_cmap(cluster_id)],
                s=20,
                alpha=0.6,
                label=f"Cluster {cluster_id}",
            )
            legend_handles.append(handle)

        plt.scatter(
            reduced_centroids[:, 0],
            reduced_centroids[:, 1],
            c="red",
            marker="X",
            s=150,
            edgecolors="black",
            label="Centroids",
        )

        plt.title(
            f"K-means clustering with k={current_k}, "
            f"silhouette={score:.3f}"
        )
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(title="K-means")
        plt.tight_layout()
        plt.savefig(output / f"kmeans_k_g{current_k}.png")
        plt.close()

# Function to generate and observe mean between different noisy settings
def noisy_label_experiments(seed, final_model,  X_train, X_test, y_train, y_test, max_iteration = 100):
    noisy_labels = ["Asymmetric", "Symmetric"]
    results = {}
    rng = np.random.default_rng(seed)

    for noisy_setting in noisy_labels:
        accuracies = []
        
        for _ in range(max_iteration):
            repeat_seed = rng.integers(0, 1_000_000_000)
            
            noisy_model = clone(final_model)
            
            y_train_noisy = corrupt_labels(y_train, noisy_setting, int(repeat_seed))
            noisy_model.fit(X_train, y_train_noisy)
            
            noisy_predictions = noisy_model.predict(X_test)
            
            current_accuracy = accuracy_score(y_test, noisy_predictions)
            
            accuracies.append(current_accuracy)
        
        mean_accuracy = np.mean(accuracies)
        
        std_accuracy = np.std(accuracies, ddof=1)
        
        print("Results for 3.2:")

        results[noisy_setting] = {
            "accuracies": accuracies,
            "mean": mean_accuracy,
            "std": std_accuracy,
        }
        print(
            f"{noisy_setting}: "
            f"mean accuracy = {mean_accuracy:.3f}, "
            f"standard deviation = {std_accuracy:.3f}"
        )    
    

def noisy_experiments(output, seed = 0, max_iteration = 100, neighbors = 5):
    output.mkdir(parents=True, exist_ok=True)

    # Obtain data from csv file
    data = pd.read_csv('Life_Expectancy_Data.csv')

    # Filtering data to only include each countries' data from 2014
    data = data.loc[data["Year"] == 2014].copy()    

    features = ["Life expectancy", "Adult Mortality", "Schooling", "GDP"]

    label = "Status"

    model_data = data.dropna(subset = features + [label]).copy()

    X = model_data[features]
    y = model_data[label]

    print("Rows used:", len(model_data))
    print("Classes:", y.value_counts().to_dict())

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size = 0.40, stratify = y, random_state = seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train_validation, y_train_validation, test_size = 0.50, stratify = y_train_validation, random_state = seed)


    # Converting all training, validation, and testing data to numpy array for usability
    X_train_array = X_train.to_numpy()
    X_val_array = X_val.to_numpy()
    X_test_array = X_test.to_numpy()

    y_train_array = y_train.to_numpy()
    y_val_array = y_val.to_numpy()
    y_test_array = y_test.to_numpy()

    X_final = np.vstack([X_train_array, X_val_array])
    y_final = np.concatenate([y_train_array, y_val_array])

    # Generate initial SVC to find optimal C and gamma
    pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("svc", SVC(kernel="rbf")),
    ])

    parameter_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
    }

    search = GridSearchCV(
    pipeline,
    parameter_grid,
    cv=5,
    scoring="balanced_accuracy",
    )

    search.fit(X_train_array, y_train_array)
    val_predictions = search.predict(X_val_array)

    print("Best parameters:", search.best_params_)

    
    # Use optimal C and gamma to generate a clean model for feature usability
    final_model = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=search.best_params_["svc__C"],
            gamma=search.best_params_["svc__gamma"],
        )),
    ])
    
    final_model.fit(X_final, y_final)

    # Baseline has been established
    test_predictions = final_model.predict(X_test_array)
    
    # Generate state with inputted seed    
    rng = np.random.default_rng(seed)

    # Print noisy experiment for 3.2 with multiple seeds 
    noisy_label_experiments(
        seed,
        final_model,
        X_train_array,
        X_test_array,
        y_train_array,
        y_test_array,
        max_iteration,
    )
        
    # Used StandardScaler to scale all values in each feature, since GDP (one of the features) could outweigh all other features.
    neighbor_scaler = StandardScaler()

    X_train_scaled = neighbor_scaler.fit_transform(X_train_array)
     
    
    noisy_labels = ["Asymmetric", "Symmetric"]
    experiment_results = {
    noise_setting: {
        "noisy": [],
        "repaired": [],
    }
    for noise_setting in noisy_labels
    }
    

    # Create clean model for 3.4 
    clean_model = clone(final_model)
    clean_model.fit(X_final, y_final)

    clean_predictions = clean_model.predict(X_test_array)
    clean_accuracy = accuracy_score(y_test_array, clean_predictions)
    print("Clean test accuracy:", clean_accuracy)

    experiment_rows = []
    
    # Main loop for 3.4, generates classifier for clean, noisy, and repaired labels
    for noise_setting in noisy_labels:
        for iteration, repeat_seed in enumerate(
            rng.integers(0, 1_000_000, size=max_iteration),
            start=1,
        ):
            repeat_seed = int(repeat_seed)

        
            y_train_noisy = corrupt_labels(
            y_train_array,
            noise_setting,
            repeat_seed,
            )

            y_noisy_final = np.concatenate([
            np.asarray(y_train_noisy),
            y_val_array,
            ])

            # Clone models to maintain sizing of original model
            noisy_model = clone(final_model)
            noisy_model.fit(X_final, y_noisy_final)

            y_train_repaired = repair_labels(X_train_scaled, y_train_noisy, neighbors)


            y_repaired_final = np.concatenate([
                np.asarray(y_train_repaired),
                y_val_array,
            ])

            repair_model = clone(final_model)
            repair_model.fit(X_final, y_repaired_final)

            noisy_predictions = noisy_model.predict(X_test_array)
            repair_predictions = repair_model.predict(X_test_array)

            noisy_accuracy = accuracy_score(y_test_array, noisy_predictions)
            repair_accuracy = accuracy_score(y_test_array, repair_predictions)

            experiment_results[noise_setting]["noisy"].append(
            noisy_accuracy
            )
            experiment_results[noise_setting]["repaired"].append(
                repair_accuracy
            )

            experiment_rows.append({
                "noise_setting": noise_setting,
                "iteration": iteration,
                "repeat_seed": repeat_seed,
                "clean_accuracy": clean_accuracy,
                "noisy_accuracy": noisy_accuracy,
                "repaired_accuracy": repair_accuracy,
            })

    experiment_csv = output / "noisy_label_comparison.csv"
    pd.DataFrame(experiment_rows).to_csv(
        experiment_csv,
        index=False,
    )
    print(f"Saved iteration results to: {experiment_csv}")

    print("Results for 3.4:")
    print(
    f"{'Setting':<12}"
    f"{'Noisy mean':>12}"
    f"{'Noisy std':>12}"
    f"{'Repair mean':>13}"
    f"{'Repair std':>12}"
    )

    for noise_setting in noisy_labels:
        noisy_scores = experiment_results[noise_setting]["noisy"]
        repair_scores = experiment_results[noise_setting]["repaired"]

        print(
            f"{noise_setting:<12}"
            f"{np.mean(noisy_scores):>12.3f}"
            f"{np.std(noisy_scores, ddof=1):>12.3f}"
            f"{np.mean(repair_scores):>13.3f}"
            f"{np.std(repair_scores, ddof=1):>12.3f}"
        )
                    

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type = int, default = 0)
    
    parser.add_argument("--kmeans_output", type = Path, default = Path(__file__).resolve().parent / "results" / "kmeans",)
    parser.add_argument("--noisy_output", type = Path, default = Path(__file__).resolve().parent / "results" / "noisy",)

    parser.add_argument("--kmeans", type = bool, default = False)

    parser.add_argument("--noisy", type = bool, default = False)
    
    parser.add_argument("--neighbors", type = int, default = 5)
    
    args = parser.parse_args()
    
    if args.noisy == True:
        noisy_experiments(args.noisy_output, args.seed, args.repeats, args.neighbors)
    
    if args.kmeans == True:
        kmeans_experiments(args.kmeans_output, args.seed, args.repeats)
    
    
    



if __name__ == "__main__":
    main()
    
