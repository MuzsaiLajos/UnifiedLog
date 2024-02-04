#!/bin/bash

# Source directory containing .pkl files
source_dir="/home/mlajos/UnifiedLog/framework/data_preprocess/preprocessed_ascii_spec_0_9_spec_10m_bigger_token_size_1002/tokenized/"

# Destination directory for hard links
dest_dir="/home/mlajos/UnifiedLog/framework/data_preprocess/preprocessed_ascii_spec_0_9_spec_10m_bigger_token_size_1002/tokenized_no_bgl/"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop through all .pkl files in the source directory
for file in "$source_dir"*.pkl; do
    # Extract the filename without the path
    filename=$(basename "$file")

    # Create hard link in the destination directory
    ln "$file" "$dest_dir$link_prefix$filename"
done
