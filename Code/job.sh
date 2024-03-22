#!/usr/bin/env bash
#SBATCH --job-name=dask_test
#SBATCH --partition=modi_HPPC
#SBATCH --nodes=2
#SBATCH --ntasks=4
##SBATCH --exclusive

mpiexec apptainer exec \
   ~/modi_images/ucphhpc/hpc-notebook:latest \
   ~/modi_mount/tf/bin/python task_farm_HEP_dask_futures_mpi.py
