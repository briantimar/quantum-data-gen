#!/bin/bash
#SBATCH --job-name=sampling
#SBATCH --nodes=20
#SBATCH --ntasks=20
#SBATCH --output=/home/btimar/data/python-multiproc.out
#SBATCH --time=0:01:00

python sampling_config.py

for L in 2; do
    for statename in "ghz"; do
        echo "Sampling from state $statename, L=$L"
        python sample_state.py $L $statename
    done
done
    
