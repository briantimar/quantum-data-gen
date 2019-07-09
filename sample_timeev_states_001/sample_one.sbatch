#!/bin/bash
# SBATCH --time=24:00:00
# 

L=$1
tindex=$2
N=$3
seed=$4

date
echo "beginning sampling"
python sample_one.py $L $tindex $N $seed
echo "ending sampling"
date