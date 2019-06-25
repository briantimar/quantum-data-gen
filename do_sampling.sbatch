#!/bin/bash
#SBATCH --job-name=sampling
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=2
#SBATCH --output=/home/btimar/data/sampling.out
#SBATCH --time=17:00:00

date
module load python
python sampling_config.py

for L in 2 4 6 8 10; do
    for statename in "tfim_ground" "rydberg_ground" "heisenberg_ground" "ghz"; do
        echo "Sampling from state $statename, L=$L"
        srun -N 1 -n 1 python sample_state.py $L $statename &
    done
done
wait
date
