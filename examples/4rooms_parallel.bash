 #!/bin/bash

dim=14
duration=60000000
repetitions=2
init_seed=114364114
rmax=1 #${dim}

folder=rooms_${dim}_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}

N_PARALLEL="10"
COMMAND="python ../example_navgrid.py"
ALGS="FSUCRLv1 FSUCRLv2 SUCRL UCRL"
opt_alpha=" --p_alpha 0.02 --mc_alpha 0.02 --r_alpha 0.8 --tau_alpha 0.8 "
fix_options=" -d ${dim} -n ${duration} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} --id c1"

cd $folder
export OMP_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
parallel -j $N_PARALLEL $COMMAND ${opt_alpha} ${fix_options} --alg ${1} ::: ${ALGS}
