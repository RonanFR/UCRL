#!/bin/bash
folder=rooms_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}

N_cpu=12
N_mem=24G

dim=18
duration=60000000
repetitions=5
init_seed=114364114
rmax=${dim}
tmax=${dim} #$((1+dim/2))

# CREATE CONFIGURATION FOR FSUCRL

#v1
i=1
sname=fsucrlv1_${dim}.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=r_fv1_${dim}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=fsucrlv1_${dim}_%j.out" >> ${fname}
echo "pwd; hostname; date" >> ${fname}
echo "" >> ${fname}
echo "module load anaconda3/4.1.0" >> ${fname}
echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

echo "python ../example_roommaze.py -b --alg FSUCRLv1 -d ${dim} -n ${duration} --tmax ${tmax} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname} #bernstein
i=$((i+1))
echo "python ../example_roommaze.py --alg FSUCRLv1 -d ${dim} -n ${duration} --tmax ${tmax} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

cd ${folder}
sbatch ${sname}
cd ..
sleep 1

#v2
i=1
sname=fsucrlv2_${dim}.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=fv2_${dim}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=fsucrlv2_${dim}_%j.out" >> ${fname}
echo "pwd; hostname; date" >> ${fname}
echo "" >> ${fname}
echo "module load anaconda3/4.1.0" >> ${fname}
echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

echo "python ../example_roommaze.py -b --alg FSUCRLv2 -d ${dim} -n ${duration} --tmax ${tmax} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname} #bernstein
i=$((i+1))
echo "python ../example_roommaze.py --alg FSUCRLv2 -d ${dim} -n ${duration} --tmax ${tmax} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

cd ${folder}
sbatch ${sname}
cd ..
sleep 1

# CREATE CONFIGURATION FOR SUCRL
i=1
sname=sucrl_${dim}.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=sucrl_${dim}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=sucrl_${dim}_%j.out" >> ${fname}
echo "pwd; hostname; date" >> ${fname}
echo "" >> ${fname}
echo "module load anaconda3/4.1.0" >> ${fname}
echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

echo "python ../example_roommaze.py -b --alg SUCRL -d ${dim} -n ${duration} --tmax ${tmax} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname} #bernstein
i=$((i+1))
echo "python ../example_roommaze.py --alg SUCRL -d ${dim} -n ${duration} --tmax ${tmax} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

cd ${folder}
sbatch ${sname}
cd ..
sleep 1

# CREATE CONFIGURATION FOR MDP
i=1
sname=ucrl_${dim}.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=ucrl_${dim}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=ucrl_${dim}_%j.out" >> ${fname}
echo "pwd; hostname; date" >> ${fname}
echo "" >> ${fname}
echo "module load anaconda3/4.1.0" >> ${fname}
echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

echo "python ../example_roommaze.py -b --alg UCRL -d ${dim} -n ${duration} --tmax ${tmax} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname} #bernstein
i=$((i+1))
echo "python ../example_roommaze.py --alg UCRL -d ${dim} -n ${duration} --tmax ${tmax} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

cd ${folder}
sbatch ${sname}
cd ..
sleep 1
