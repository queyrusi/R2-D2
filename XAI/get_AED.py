#!/usr/bin/env python
"""
This script calculates the Average Euclidean Distance (AED) between heatmaps.
"""

import glob
import csv
import os
import argparse
import numpy as np
from tqdm import tqdm

def eucl_dist(h1, h2):
    return np.linalg.norm(h1 - h2)

def save_result(f, label, value):
    # Save the result to a file or perform any other desired action
    f.write(f"{label}: {float(value)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D0new", help="Dataset name")
    args = parser.parse_args()

    dataset = args.dataset
    path_to_gists = f"gists/{dataset}/*.csv"

    # Load gist vectors
    vectors = []
    csv_files = glob.glob(path_to_gists)  # Replace with the actual path to your CSV files

    for file in tqdm(csv_files, desc='getting vectors from gists'):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the first line
            data = next(reader)
            vector = np.array([float(num) for num in data])
            vectors.append(vector)

    num_vectors = len(vectors)

    # Initialize working variables
    sum_ED = 0  # Sum of Euclidean Distances
    dist_01 = 0  # Distances < 0.1
    dist_02 = 0  # 0.1 <= distances < 0.2

    # Compare heatmaps with each other
    for h_base in tqdm(vectors, desc='comparing heatmaps'):
        # Initialize working variables
        temp_eucl_sum = 0

        for h in vectors:
            # Calculate Euclidean distance
            temp_dist = eucl_dist(h_base, h)

            # Update counters
            temp_eucl_sum += temp_dist

            if temp_dist < 0.1:
                dist_01 += 1
            elif 0.1 <= temp_dist < 0.2:
                dist_02 += 1

        sum_ED += temp_eucl_sum / num_vectors

        results_dir = f'results/{dataset}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            open(f'{results_dir}/AED.txt', 'a').close()

    with open(f'results/{dataset}/AED.txt', 'w') as f:
        save_result(f, 'AED', sum_ED / num_vectors)  # Average ED
        save_result(f, '<0.1', dist_01 / (num_vectors * num_vectors))
        save_result(f, '0.1-0.2', dist_02 / (num_vectors * num_vectors))