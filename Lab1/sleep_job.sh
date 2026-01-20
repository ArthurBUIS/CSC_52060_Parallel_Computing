#!/bin/bash
#SBATCH --job-name=sleep_test
#SBATCH --output=slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

sleep 1h
