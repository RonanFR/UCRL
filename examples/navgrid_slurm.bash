#!/bin/bash
N_cpu=16
N_mem=50g

dim=20
duration=80000000
repetitions=1
init_seed=114364114
rmax=${dim}
tmax=${dim} #$((1+dim/2))


folder=navgrid_${dim}_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}

# CREATE CONFIGURATION FOR FSUCRL

#v1
i=1
for (( t=1; t<=${tmax}; t++ ))
do
    sname=fsucrlv1.slurm
    fname=${folder}/${sname}
    echo "#!/bin/bash" > ${fname}                                                                                                      
    echo "#SBATCH --nodes=1" >> ${fname}
    echo "#SBATCH --ntasks-per-node=1" >> ${fname}
    echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
    echo "#SBATCH --time=24:00:00" >> ${fname}
    echo "#SBATCH --job-name=gf1_${dim}_${t}" >> ${fname}
    echo "#SBATCH --mem=${N_mem}" >> ${fname}
    echo "#SBATCH --output=fsucrlv1_${dim}_${t}_%j.out" >> ${fname}
    echo "pwd; hostname; date" >> ${fname}
    echo "" >> ${fname}
    echo "module load anaconda3/4.1.0" >> ${fname}
    echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

    echo "python ../example_navgrid.py --alg FSUCRLv1 -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
    i=$((i+1))
    echo "python ../example_navgrid.py -b --alg FSUCRLv1 -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

    cd ${folder}
    sbatch ${sname}
    cd ..
    sleep 1
done

#v2
i=1
for (( t=1; t<=${tmax}; t++ ))
do
    sname=fsucrlv2.slurm
    fname=${folder}/${sname}
    echo "#!/bin/bash" > ${fname}                                                                                                      
    echo "#SBATCH --nodes=1" >> ${fname}
    echo "#SBATCH --ntasks-per-node=1" >> ${fname}
    echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
    echo "#SBATCH --time=24:00:00" >> ${fname}
    echo "#SBATCH --job-name=gf2_${dim}_${t}" >> ${fname}
    echo "#SBATCH --mem=${N_mem}" >> ${fname}
    echo "#SBATCH --output=fsucrlv2_${dim}_${t}_%j.out" >> ${fname}
    echo "pwd; hostname; date" >> ${fname}
    echo "" >> ${fname}
    echo "module load anaconda3/4.1.0" >> ${fname}
    echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

    echo "python ../example_navgrid.py --alg FSUCRLv2 -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
    i=$((i+1))
    echo "python ../example_navgrid.py -b --alg FSUCRLv2 -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

    cd ${folder}
    sbatch ${sname}
    cd ..
    sleep 1
done

# CREATE CONFIGURATION FOR SUCRL
i=1
for (( t=1; t<=${tmax}; t++ ))
do
    sname=sucrl.slurm
    fname=${folder}/${sname}
    echo "#!/bin/bash" > ${fname}                                                                                                      
    echo "#SBATCH --nodes=1" >> ${fname}
    echo "#SBATCH --ntasks-per-node=1" >> ${fname}
    echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
    echo "#SBATCH --time=24:00:00" >> ${fname}
    echo "#SBATCH --job-name=gsu_${dim}_${t}" >> ${fname}
    echo "#SBATCH --mem=${N_mem}" >> ${fname}
    echo "#SBATCH --output=sucrl_${dim}_${t}_%j.out" >> ${fname}
    echo "pwd; hostname; date" >> ${fname}
    echo "" >> ${fname}
    echo "module load anaconda3/4.1.0" >> ${fname}
    echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

    echo "python ../example_navgrid.py --alg SUCRL -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
    i=$((i+1))
    echo "python ../example_navgrid.py -b --alg SUCRL -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

    cd ${folder}
    sbatch ${sname}
    cd ..
    sleep 1
done

# CREATE CONFIGURATION FOR MDP
i=1
sname=ucrl.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=guc_${dim}" >> ${fname}
echo "#SBATCH --mem=${N_mem}" >> ${fname}
echo "#SBATCH --output=mdpucrl_${dim}_%j.out" >> ${fname}
echo "pwd; hostname; date" >> ${fname}
echo "" >> ${fname}
echo "module load anaconda3/4.1.0" >> ${fname}
echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

echo "python ../example_roommaze.py --alg UCRL -d ${dim} -n ${duration} --tmax -1 --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
i=$((i+1))
echo "python ../example_roommaze.py -b --alg UCRL -d ${dim} -n ${duration} --tmax -1 --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

cd ${folder}
sbatch ${sname}
cd ..
sleep 1
