#!/bin/bash
# Run all Python scripts in the scripts directory using uv. Prior to running
# this script, ensure you have uv installed and that you are in the correct
# directory where the contents folder is located. Run `uv sync` to install any
# dependencies before executing this script.
#
# Made by Léonard Seydoux, Feb. 2026

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
failed=()

run_scripts() {
    local label="$1"
    local dir="$2"
    local runner="$3"  # "uv" or "python"

    echo ""
    echo "=== $label ==="
    for script in "$dir"/*.py; do
        [ -f "$script" ] || continue
        name="$(basename "$script")"
        printf "  %-45s" "$name..."
        if [ "$runner" = "uv" ]; then
            # Run from SCRIPT_DIR so that matplotlibrc and pyproject.toml are found
            output=$(cd "$SCRIPT_DIR" && uv run "scripts/$name" 2>&1) && rc=0 || rc=$?
        else
            # Run from the script's directory so its local matplotlibrc is found
            output=$(cd "$dir" && "$PYTHON" "$name" 2>&1) && rc=0 || rc=$?
        fi
        if [ $rc -eq 0 ]; then
            echo "OK"
        else
            echo "FAILED"
            echo "$output" | grep -v "UserWarning\|cmr10\|findfont" | tail -5 | sed 's/^/    /'
            failed+=("$script")
        fi
    done
}

run_scripts "Contents figures"    "$SCRIPT_DIR/scripts"             "uv"
run_scripts "Supervised figures"  "$SCRIPT_DIR/../images/supervised" "python"
run_scripts "Unsupervised figures" "$SCRIPT_DIR/../images/unsupervised" "python"

echo ""
if [ ${#failed[@]} -eq 0 ]; then
    echo "All figures generated successfully."
else
    echo "${#failed[@]} script(s) failed:"
    for f in "${failed[@]}"; do echo "  - $f"; done
    exit 1
fi