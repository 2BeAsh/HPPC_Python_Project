#!/usr/bin/env bash
#SBATCH --job-name=mpi4py
#SBATCH --partition=modi_HPPC
#SBATCH --nodes=1
#SBATCH --ntasks=1

mpirun apptainer exec \
 ~/modi_images/ucphhpc/hpc-notebook:latest \
 ~/modi_mount/tf/bin/python dask_test.py