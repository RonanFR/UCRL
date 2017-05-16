
run_all: clean run_freesmdp run_smdp run_mdp

run_freesmdp:
	        sbatch freesmdp.slurm 

run_smdp:
	        sbatch smdp.slurm

run_mdp:
	        sbatch mdp.slurm

tar_all:
	        tar cvzf "res_`date '+%Y%m%d_%H%M%S'`.tar.gz" */ *.out

clean: tar_all
	        rm -rf */ *.out

