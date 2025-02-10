#!/bin/bash

## Run the python script sgpt_file.py before proceeding
#echo "Running python sgpt_file.py..."
#python sgpt_file.py
#
## Check if the python script ran successfully
#if [ $? -ne 0 ]; then
#    echo "Error: Failed to run sgpt_file.py"
#    exit 1
#fi

echo "Current working directory: $(pwd)"
base_dir="./config/Llama2-7b-chat"
echo "Base directory: $base_dir"

cd "$(dirname "$0")" || exit 1

retrievers=("SGPT" "BM25" "SBERT")

for subdir in "$base_dir"/*/; do
    if [ -d "$subdir" ]; then
        echo "Checking directory: $subdir"
        config_path="$subdir/DioR.json"

        if [ -f "$config_path" ]; then
            echo "Found config file: $config_path"

            for retriever in "${retrievers[@]}"; do
                echo "Setting retriever to $retriever in $config_path"

                python - <<EOF
import json
import sys

config_path = "$config_path"
retriever = "$retriever"

try:
    with open(config_path, "r") as f:
        data = json.load(f)

    data["retriever"] = retriever

    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Successfully updated retriever to {retriever} in {config_path}")
except Exception as e:
    print(f"Failed to update retriever in {config_path}: {e}", file=sys.stderr)
    sys.exit(1)
EOF

                if [ $? -ne 0 ]; then
                    echo "Error modifying retriever in $config_path with value $retriever"
                    continue
                fi

                python src/main.py -c "$config_path"

                if [ $? -ne 0 ]; then
                    echo "Error running python script for $config_path with retriever $retriever"
                fi
            done
        else
            echo "No DioR.json found in $subdir"
        fi
    else
        echo "Skipping non-directory entry: $subdir"
    fi
done
