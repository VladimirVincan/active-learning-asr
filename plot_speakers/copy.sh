#!/bin/bash

# Specify the source directory where your JSON files are located

# Specify the destination directory where you want to copy the files

# Specify the new extension for the copied files
new_extension=".txt"

# Iterate over all files ending with .json in the source directory
for file in *.json; do
    # Extract the filename (without extension) from the full path
    filename=$(basename "$file" .json)

    # Copy the file to the destination directory with the new extension
    cp "$file" "$filename.json.sample"
done
