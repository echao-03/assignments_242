import argparse
from pathlib import Path

import csv


BASE_DIR = Path(__file__).resolve().parent
K_CLUSTERS = 10
PCA_CSV = BASE_DIR / "results" / "kmeans" / f"reduced_pca_k{K_CLUSTERS}.csv"
LIFE_EXPECTANCY_CSV = BASE_DIR / "Life_Expectancy_Data.csv"


def run(output):
    life_data = {}

    with LIFE_EXPECTANCY_CSV.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        reader.fieldnames = [
            field.strip() if field is not None else field
            for field in reader.fieldnames
        ]

        for row in reader:
            row = {key.strip(): value for key, value in row.items()}
            if row["Year"] == "2014":
                life_data[row["Country"]] = {
                    "life_expectancy": float(row["Life expectancy"]),
                    "adult_mortality": float(row["Adult Mortality"]),
                    "alcohol_consumption": float(row["Alcohol"]),
                    "gdp": float(row["GDP"]),
                    "status": row["Status"],
                }

    cluster_data = {str(cluster): [] for cluster in range(K_CLUSTERS)}

    with PCA_CSV.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            country = row["Country"]
            cluster = row["Cluster"]

            if country in life_data:
                cluster_data[cluster].append({
                    "country": country,
                    **life_data[country],
                })

    feature_names = (
        "life_expectancy",
        "adult_mortality",
        "alcohol_consumption",
        "gdp",
    )

    print("Cluster feature means:")
    for cluster in range(K_CLUSTERS):
        observations = cluster_data[str(cluster)]

        if not observations:
            print(f"Cluster {cluster}: no countries")
            continue

        means = {}
        for feature in feature_names:
            values = [observation[feature] for observation in observations]
            means[feature] = sum(values) / len(values)

        print(
            f"Cluster {cluster}: "
            f"life_expectancy={means['life_expectancy']:.2f}, "
            f"adult_mortality={means['adult_mortality']:.2f}, "
            f"alcohol_consumption={means['alcohol_consumption']:.2f}, "
            f"gdp={means['gdp']:.2f}"
        )

        country_status = [
            f"{observation['country']} ({observation['status']})"
            for observation in sorted(
                observations,
                key=lambda observation: observation["country"],
            )
        ]
        print(f"Countries: {', '.join(country_status)}")

    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "cluster",
        "country",
        "status",
        "life_expectancy",
        "adult_mortality",
        "alcohol_consumption",
        "gdp",
    ]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for cluster, observations in cluster_data.items():
            for observation in observations:
                writer.writerow({
                    "cluster": cluster,
                    **observation,
                })
def main():
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        choices=(5,),
        help="Use five neighbors for this assignment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "observations1.csv",
    )
    args = parser.parse_args()

    
    run(args.output)
    
    
    
if __name__ == "__main__":
    main()