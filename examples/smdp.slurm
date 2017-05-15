#!/bin/bash                                                                                                               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --job-name=ucrl_smdp
#SBATCH --mem=80G
#SBATCH --output=smdp_%j.out

pwd; hostname; date
echo "Running SMDP-UCRL program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS tasks, each with $SLURM_CPUS_PER_TASK cores."

module load anaconda3/4.1.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
python example_smdp.py -d 20 -n 30000000 --tmax 4 -c 0.8 --rmax 20 --p_range 0.1 -r 10 --seed 17632376383341475136 --id c1
python example_smdp.py -d 20 -n 30000000 --tmax 8 -c 0.8 --rmax 20 --p_range 0.1 -r 10 --seed 17632376383341475136 --id c2
python example_smdp.py -d 30 -n 50000000 --tmax 4 -c 0.8 --rmax 30 --p_range 0.1 -r 10 --seed 17632376383341475136 --id c3
python example_smdp.py -d 30 -n 50000000 --tmax 8 -c 0.8 --rmax 30 --p_range 0.1 -r 10 --seed 17632376383341475136 --id c4

