#!/bin/bash
# N_cpu=16
# N_mem=50g
N_cpu=8
N_mem=10g
N_hours=24
part=24c

dim=14
duration=160000000
N_parallel_rep=1
repetitions=5
init_seed=(114364114 86231 34556 2538764)
rmax=1 #${dim}
exe_file=../example_roommaze.py 


folder=rooms_${dim}_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}


ALGS=(FSUCRLv1 FSUCRLv2 UCRL SUCRL_v2 SUCRL_v3)
A_SHORT_NAME=(R-fv1 R-fv2 R-ucr R-2suc R-3suc)

# CREATE CONFIGURATIONS

ALPHAS=" --p_alpha 0.02 --mc_alpha 0.02 --r_alpha 0.8 --tau_alpha 0.8 "

for (( pr=0; pr<${N_parallel_rep}; pr++ ))
do
    for (( j=0; j<${#ALGS[@]}; j++ ))
    do
        echo ${j} ${ALGS[$j]}
        
        i=1
        out_name="${ALGS[$j]}_${pr}_${dim}_%j.out"
        sname=${ALGS[$j]}_${dim}.slurm
        fname=${folder}/${sname}
        
        echo "#!/bin/bash" > ${fname}                                                                                                      
        echo "#SBATCH --nodes=1" >> ${fname}
        echo "#SBATCH --partition=${part}" >> ${fname}
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
        
        #cmdp=" --id c${i}"
        cmdp=" --path ${ALGS[$j]}_4rooms_c${i}"
        
        echo "python ${exe_file} --alg ${ALGS[$j]} ${ALPHAS} -d ${dim} -n ${duration} --rmax ${rmax} -r ${repetitions} --seed ${init_seed[$pr]} ${cmdp}" >> ${fname}
        i=$((i+1))
        # echo "python ${exe_file} -b --alg ${ALGS[$j]} ${ALPHAS} -d ${dim} -n ${duration} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
        # i=$((i+1))

        cd ${folder}
        sbatch ${sname}
        cd ..
        sleep 1

    done
done
