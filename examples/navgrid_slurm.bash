#!/bin/bash
# N_cpu=16
# N_mem=50g
N_cpu=8
N_mem=10g
N_hours=24
part=24c

dim=20
duration=120000000
N_parallel_rep=1
repetitions=1
init_seed=(114364114 679848179 375341576 340061651 311346802 945527102 1028531057 358887046 299813034 472903536 650815502 931560826 391431306 111281634 55536093 484610172 131932607 835579495 82081514 603410165 467299485)
rmax=1 #${dim}
tmax=$((dim/2+2))
exe_file=../example_navgrid.py 
bound_type="chernoff"


folder=navgrid_${dim}_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}


ALGS=(FSUCRLv1 FSUCRLv2 UCRL SUCRL_v1 SUCRL_v2 SUCRL_v3 SUCRL_v4 SUCRL_v5)
A_SHORT_NAME=(N-fv1 N-fv2 N-ucr N-1suc N-2suc N-3suc N-4suc N-5suc)

# CREATE CONFIGURATIONS

ALPHAS=" --p_alpha 0.02 --mc_alpha 0.02 --r_alpha 0.8 --tau_alpha 0.8 "

for (( pr=0; pr<${N_parallel_rep}; pr++ ))
do
    off=$((pr*repetitions))
    for (( j=0; j<${#ALGS[@]}; j++ ))
    do
        echo ${pr} ${j} ${ALGS[$j]}
        N_t=${tmax}
        if [ ${ALGS[$j]} == UCRL ]
        then
            N_t=1
        fi
        i=1
        for (( t=1; t<=${N_t}; t++ ))
        do
            out_name="${ALGS[$j]}_${pr}_${dim}_${t}_%j.out"
            sname=${ALGS[$j]}_${pr}_${dim}_${t}.slurm
            fname=${folder}/${sname}
            
            echo "#!/bin/bash" > ${fname}                                                                                                      
            echo "#SBATCH --nodes=1" >> ${fname}
            echo "#SBATCH --ntasks-per-node=1" >> ${fname}
            echo "#SBATCH --partition=${part}" >> ${fname}
            echo "#SBATCH --cpus-per-task=${N_cpu}" >> ${fname}
            echo "#SBATCH --time=${N_hours}:00:00" >> ${fname}
            echo "#SBATCH --job-name=${A_SHORT_NAME[$j]}_${dim}_${t}" >> ${fname}
            echo "#SBATCH --mem=${N_mem}" >> ${fname}
            echo "#SBATCH --output=${out_name}" >> ${fname}
            echo "pwd; hostname; date" >> ${fname}
            echo "" >> ${fname}
            echo "module load anaconda3/4.1.0" >> ${fname}
            echo "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
            echo "export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK" >> ${fname}
            
            #cmdp=" --id c${i}"
            cmdp="--rep_offset ${off} --path ${ALGS[$j]}_navgrid_c${i}"

            echo "python ${exe_file} -b ${bound_type} --alg ${ALGS[$j]} ${ALPHAS} -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed[$pr]} ${cmdp}" >> ${fname}
            i=$((i+1))
            
            # cmdp="--rep_offset ${off} --path ${ALGS[$j]}_4rooms_c${i}"
            # echo "python ${exe_file} -b --alg ${ALGS[$j]} ${ALPHAS} -d ${dim} -n ${duration} --tmax ${t} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c${i}" >> ${fname}
            # i=$((i+1))

            cd ${folder}
            sbatch ${sname}
            cd ..
            sleep 1

        done
    done
done
