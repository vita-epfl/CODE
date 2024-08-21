#!/bin/bash

# List of strings
strings=("sde_without_clipping" "ode_with_clipping" "ode_without_clipping" "corrupted"  "reconstruction_without_clipping" "reconstruction_with_clipping" )

# Define the value of n
n=$(nvidia-smi -L | wc -l)

# Loop over the list of strings
for ((i=0; i<${#strings[@]}; i++)); do
    # Calculate i modulo n
    mod=$((i % n))
    
    # Run each string with the argument (i modulo n)
    echo "Running ${strings[$i]} with argument $mod"
    # Replace the echo command with the actual command you want to run
    python calculate_metrics.py "${strings[$i]}" "$mod" &

    # If we've hit every n iterations, use wait
    if (( (i+1) % n == 0 )); then
        wait
    fi
done