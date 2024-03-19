#!/usr/bin/env bash
#SBATCH --job-name=TaskFarm
#SBATCH --partition=modi_HPPC
#SBATCH --nodes=1
#SBATCH --exclusive

for ntasks in {2..64}; do
    mpiexec -n $ntasks apptainer exec \
       ~/modi_images/ucphhpc/hpc-notebook:latest \
       ~/modi_mount/tf/bin/python task_farm_HEP_bunch.py 
done
