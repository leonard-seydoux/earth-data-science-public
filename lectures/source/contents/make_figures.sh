#!/bin/bash
# Run all Python scripts in the scripts directory using uv Prior to running this
# script, ensure you have uv installed and that you are in the correct directory
# where the contents folder is located. Run `uv sync` to install any
# dependencies before executing this script.
#
# Made by Léonard Seydoux, Feb. 2026

echo `pwd`

for script in scripts/*.py; do
    echo "Running $script..."
    uv run "$script"
donerun