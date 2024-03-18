import sys
sys.path.append("/home/jovyan/erda_mount/__dag_config__/python3")
# from mpi4py import MPI
import numpy as np
import time
from overall import Data, task_function, set_gen

n_cuts = 3
n_settings = n_cuts ** 8

def master(mpi_size,ds):
    print(f'I am the master! I have {mpi_size-1} workers')
    print(f'Nsig = {ds.nsig}, Nbkg = {ds.nbckg}, Ntot = {ds.nevents}')
    settings = set_gen(ds, n_cuts, n_settings)
    
    # loop over different event channels and set up cuts


    accuracy = np.zeros(n_settings)

    #timer start
    start_time = time.time()

    for i in range(n_settings):
        accuracy[i] = task_function(settings[i,:],ds)
    
    #timer stop
    stop_time = time.time()
        
    best_accuracy_score = np.max(accuracy)
    best_accuracy_setting = settings[np.argmax(accuracy)]

    print(f'Best accuracy optained: {best_accuracy_score:.6f}')
    print('Final cuts:')
    for i in range(8):
        print(ds.name[i] + f': {ds.flip[i]*best_accuracy_setting[i]:.6f}')

    print(f'Number of settings: {n_settings}')
    print(f'Elapsed time: {stop_time-start_time:.2f} seconds')


def worker(rank,ds):
    print(f'i am worker {rank}')




# main:
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
rank = 0
data_time_start = time.time()
Filename = 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
ds = Data(Filename)
data_time_stop = time.time()
print(f'data loading time: {data_time_stop - data_time_start:.2f} seconds')

if rank == 0:
    # comm.send(data, dest=1, tag=11)
    # mpi_size = comm.Get_size()
    mpi_size = 1
    master(mpi_size,ds)
else:
    # data = comm.recv(source=0, tag=11)
    worker(rank,ds)
