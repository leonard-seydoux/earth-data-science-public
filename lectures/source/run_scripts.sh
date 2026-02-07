#!/bin/bash

for script in contents/scripts/*.py; do
    echo "Running $script"
    uv run "$script"
done
