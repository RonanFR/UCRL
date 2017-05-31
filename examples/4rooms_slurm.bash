#!/bin/bash
N_cpu=16
N_mem=50g
N_hours=24

dim=18
duration=60000000
repetitions=1
init_seed=114364114
rmax=${dim}
exe_file=../example_roommaze.py 


folder=rooms_${dim}_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}


ALGS=(FSUCRLv1 FSUCRLv2 SUCRL UCRL)
A_SHORT_NAME=(R-fv1 R-fv2 R-suc R-ucr)

# CREATE CONFIGURATIONS

for (( j=0; j<${#ALGS[@]}; j++ ))
do
    echo ${j} ${ALGS[$j]}
    
    i=1
    out_name="${ALGS[$j]}_${dim}_%j.out"
    sname=${ALGS[$j]}_${dim}.slurm
    fname=${folder}/${sname}
    
    echo "#!/bin/bash" > ${fname}                                                                                                      
    echo "#SBATCH --nodes=1" >> ${fname}
    echo "#SBATCH --ntasks-per-node=1" >> ${fname}
    echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
    echo "#SBATCH --time=${N_hours}:00:00" >> ${fname}
    echo "#SBATCH --job-name=${A_SHORT_NAME[$j]}_${dim}" >> ${fname}
    echo "#SBATCH --mem=${N_mem}" >> ${fname}
    echo "#SBATCH --output=${out_name}" >> ${fname}
    echo "pwd; hostname; date" >> ${fname}
    echo "" >> ${fname}
    echo "module load anaconda3/4.1.0" >> ${fname}
    echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
    echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}

    echo "python ${exe_file} --alg ${ALGS[$j]} -d ${dim} -n ${duration} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
    i=$((i+1))
    echo "python ${exe_file} -b --alg ${ALGS[$j]} -d ${dim} -n ${duration} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}

    cd ${folder}
    sbatch ${sname}
    cd ..
    sleep 1

done
