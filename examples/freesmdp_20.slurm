#!/bin/bash                                                                                                               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --job-name=ucrl_freesmdp
#SBATCH --mem=80G
#SBATCH --output=freesmdp_%j.out

pwd; hostname; date
echo "Running FREESMDP-UCRL program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS tasks, each with $SLURM_CPUS_PER_TASK cores."

module load anaconda3/4.1.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
python example_free_smdp.py -d 20 -n 30000000 --tmax 4  --rmax 20 -r 3 --seed 17632376383341475136 --id c1
python example_free_smdp.py -d 20 -n 30000000 --tmax 8  --rmax 20 -r 3 --seed 17632376383341475136 --id c2
python example_free_smdp.py -d 20 -n 30000000 --tmax 16 --rmax 20 -r 3 --seed 17632376383341475136 --id c3
