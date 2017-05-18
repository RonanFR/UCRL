#!/bin/bash
folder=test_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}


dim=20
duration=40000000
tmax=8
repetitions=5
init_seed=28632376383341475136
rmax=${dim}

# CREATE CONFIGURATION FOR FSUCRL

#v1
i=1
for (( t=1; t<=${tmax}; t++ ))
do
    sname=fsucrl1_${dim}_${t}.slurm
    fname=${folder}/${sname}
    echo "#!/bin/bash" > ${fname}                                                                                                      
    echo "#SBATCH --nodes=1" >> ${fname}
    echo "#SBATCH --ntasks-per-node=1" >> ${fname}
    echo "#SBATCH --cpus-per-task=24" >> ${fname}
    echo "#SBATCH --time=24:00:00" >> ${fname}
    echo "#SBATCH --job-name=fsucrl1_${dim}_${t}" >> ${fname}
    echo "#SBATCH --mem=24G" >> ${fname}
    echo "#SBATCH --output=fsucrl1_${dim}_${t}_%j.out" >> ${fname}
    echo "pwd; hostname; date" >> ${fname}
    echo "" >> ${fname}
    echo "module load anaconda3/4.1.0" >> ${fname}
    echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    
    echo "python ../example_free_smdp.py --v_alg 1 -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
    i=$((i+1))
    echo "python ../example_free_smdp.py -b --v_alg 1 -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname} #bernstein
    i=$((i+1))
    
    cd ${folder}
    sbatch ${sname}
    cd ..
    sleep 1
done

#v2
i=1
for (( t=1; t<=${tmax}; t++ ))
do
    sname=fsucrl1_${dim}_${t}.slurm
    fname=${folder}/${sname}
    echo "#!/bin/bash" > ${fname}                                                                                                      
    echo "#SBATCH --nodes=1" >> ${fname}
    echo "#SBATCH --ntasks-per-node=1" >> ${fname}
    echo "#SBATCH --cpus-per-task=24" >> ${fname}
    echo "#SBATCH --time=24:00:00" >> ${fname}
    echo "#SBATCH --job-name=fsucrl1_${dim}_${t}" >> ${fname}
    echo "#SBATCH --mem=24G" >> ${fname}
    echo "#SBATCH --output=fsucrl1_${dim}_${t}_%j.out" >> ${fname}
    echo "pwd; hostname; date" >> ${fname}
    echo "" >> ${fname}
    echo "module load anaconda3/4.1.0" >> ${fname}
    echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    
    echo "python ../example_free_smdp.py --v_alg 2 -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
    i=$((i+1))
    echo "python ../example_free_smdp.py -b --v_alg 2 -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname} #bernstein
    i=$((i+1))
    
    cd ${folder}
    sbatch ${sname}
    cd ..
    sleep 1
done

# CREATE CONFIGURATION FOR SUCRL
i=1
for (( t=1; t<=${tmax}; t++ ))
do
    sname=sucrl_${dim}_${t}.slurm
    fname=${folder}/${sname}
    echo "#!/bin/bash" > ${fname}                                                                                                      
    echo "#SBATCH --nodes=1" >> ${fname}
    echo "#SBATCH --ntasks-per-node=1" >> ${fname}
    echo "#SBATCH --cpus-per-task=24" >> ${fname}
    echo "#SBATCH --time=24:00:00" >> ${fname}
    echo "#SBATCH --job-name=sucrl_${dim}_${t}" >> ${fname}
    echo "#SBATCH --mem=24G" >> ${fname}
    echo "#SBATCH --output=sucrl_${dim}_${t}_%j.out" >> ${fname}
    echo "pwd; hostname; date" >> ${fname}
    echo "" >> ${fname}
    echo "module load anaconda3/4.1.0" >> ${fname}
    echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    
    echo "python ../example_smdp.py -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
    i=$((i+1))
    echo "python ../example_smdp.py -b -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname} #bernstein
    i=$((i+1))
    
    cd ${folder}
    sbatch ${sname}
    cd ..
    sleep 1
done

# CREATE CONFIGURATION FOR MDP
sname=mdpucrl_${dim}_${t}.slurm
fname=${folder}/${sname}
echo "#!/bin/bash" > ${fname}                                                                                                      
echo "#SBATCH --nodes=1" >> ${fname}
echo "#SBATCH --ntasks-per-node=1" >> ${fname}
echo "#SBATCH --cpus-per-task=24" >> ${fname}
echo "#SBATCH --time=24:00:00" >> ${fname}
echo "#SBATCH --job-name=mdpucrl_${dim}_${t}" >> ${fname}
echo "#SBATCH --mem=24G" >> ${fname}
echo "#SBATCH --output=mdpucrl_${dim}_${t}_%j.out" >> ${fname}
echo "pwd; hostname; date" >> ${fname}
echo "" >> ${fname}
echo "module load anaconda3/4.1.0" >> ${fname}
echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

echo "python ../example_mdp.py -d ${dim} -n ${duration} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c1" >> ${fname}
echo "python ../example_mdp.py -b -d ${dim} -n ${duration} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c2" >> ${fname} #bernstein

cd ${folder}
sbatch ${sname}
cd ..
