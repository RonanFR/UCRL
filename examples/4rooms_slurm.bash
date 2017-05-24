#!/bin/bash
N_cpu=16
N_mem=32G

dim=18
duration=60000000
repetitions=5
init_seed=114364114
rmax=${dim}
tmax=${dim} #$((1+dim/2))


folder=rooms_${dim}_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}

# CREATE CONFIGURATION FOR FSUCRL

#v1
i=1
sname=fsucrlv1.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=fsucrlv1_${dim}_${t}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=fsucrlv1_${dim}_${t}_%j.out" >> ${fname}
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
sname=fsucrlv2.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=fsucrlv2_${dim}_${t}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=fsucrlv2_${dim}_${t}_%j.out" >> ${fname}
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
sname=sucrl.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=sucrl_${dim}_${t}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=sucrl_${dim}_${t}_%j.out" >> ${fname}
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
sname=ucrl.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=mdpucrl_${dim}_${t}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=mdpucrl_${dim}_${t}_%j.out" >> ${fname}
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
