#!/bin/bash

# Script to generate GIST features for all PNG files in a given input directory and 
# save the results in a specified output directory.

# Usage: get_all_gists.sh <input_dir> <output_dir>
#   <input_dir>   : The directory containing the input PNG files.
#   <output_dir>  : The directory where the GIST features will be saved as CSV files.

input_dir="$1"
output_dir="$2"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

for file in "$input_dir"/*.png; do
    filename=$(basename "$file")
    filename_without_extension="${filename%.*}"
    output_path="$output_dir/$filename_without_extension.csv"

    python get_gist.py --input_path "$file" --output_path "$output_path"
    wait
done