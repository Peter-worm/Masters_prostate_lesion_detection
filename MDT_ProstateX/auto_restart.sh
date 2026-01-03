#!/bin/bash

# Check if user provided an iteration limit
if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_iterations>"
    exit 1
fi

# Validate numeric input
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: iteration count must be a positive integer."
    exit 1
fi

MAX_ITERS="$5"
COUNT=0

while [ "$COUNT" -lt 4 ]; do
    echo "Iteration $((COUNT+1)) of $MAX_ITERS"

    # Add -l only on the last iteration
    EXTRA_ARG=""
    if [ $((COUNT+1)) -eq "$MAX_ITERS" ]; then
        EXTRA_ARG="-l"
    fi

    # Add --resume only after the first iteration
    RESUME_ARG=""
    if [ "$COUNT" -gt 0 ]; then
        RESUME_ARG="--resume"
    fi

    python exec.py --mode train_test \
        --exp_source experiments/exp3 \
        --exp_dir experiments/exp3 \
        $RESUME_ARG \
        --number_of_epochs 2 \
        $EXTRA_ARG

    echo "Process stopped. Restarting in 3 seconds..."
    sleep 3

    COUNT=$((COUNT+1))
done

echo "Reached $MAX_ITERS iterations. Exiting."
