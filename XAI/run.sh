#!/bin/bash

# This script runs the XAI pipeline for a given dataset.

# Usage: sh run.sh <dataset>
# Example: sh run.sh D1

dataset="$1"

python get_heatmaps.py --image_dir "../images/$1_images/malware" &&
sh get_all_gists.sh "./heatmaps/$dataset" "./gists/$dataset" &&
python get_AED.py --dataset "$dataset"