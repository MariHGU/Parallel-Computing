#!/bin/bash -l

#SBATCH --cluster=wice
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --account=lp_edu_alg_parallel_comp

module load OpenMPI GCC Boost.MPI intel
mpirun -n $SLURM_NTASKS ./gol 100 100 2 6
