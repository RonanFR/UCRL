 #!/bin/bash

dim=20
duration=40000000
repetitions=1
init_seed=114364114
rmax=1 #${dim}
tmax=8

folder=navgrid_${dim}_$(date '+%Y%m%d_%H%M%S')
echo ${folder}
mkdir ${folder}

TMAX_SEQ=$(seq 1 ${tmax})
N_PARALLEL="10"
COMMAND="python ../example_navgrid.py"
ALGS="FSUCRLv1 FSUCRLv2 SUCRL UCRL"
opt_alpha=" --p_alpha 0.02 --mc_alpha 0.02 --r_alpha 0.8 --tau_alpha 0.8 "
fix_options=" -d ${dim} -n ${duration} --rmax ${rmax} -r ${repetitions} --seed ${init_seed} "

cd $folder
export OMP_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
parallel -j $N_PARALLEL echo $COMMAND ${opt_alpha} ${fix_options} --alg {1} --tmax {2} --id c{2} ::: ${ALGS} ::: ${TMAX_SEQ}
