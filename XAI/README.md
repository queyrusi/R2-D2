# Execution Order for Scripts in XAI

Follow the steps below to execute the scripts in the correct order:

1. **get_heatmaps.py**: This is the first script you should run. It generates heatmaps for your data.

    ```bash
    python get_heatmaps.py --image_dir <input_dir>
    ```

2. **get_all_gists.sh**: After you've generated the heatmaps, run this script. It generates GIST features for all PNG files in a given input directory and saves the results in a specified output directory.

    ```bash
    get_all_gists.sh <input_dir> <output_dir>
    ```

3. **get_AED**: This is the final script to run. It performs the AED operation and outputs an `AED.txt`

    ```bash
    python get_AED.py
    ```

Make sure to run these scripts in the order specified above to ensure that each script has the data it needs from the previous script.

4. **run.sh**: [UP TO DATE] This script runs the XAI pipeline for a given dataset.

    ```bash
    sh run.sh <dataset>
    ```

    Example:
    ```bash
    sh run.sh D1
    ```
    