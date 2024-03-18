# mpiexec --mca oob_tcp_if_include lo -np 2 python3 task_farm_HEP.py
# code adapted by ida

import sys
sys.path.append("/home/jovyan/erda_mount/__dag_config__/python3")
from mpi4py import MPI
import numpy as np
import time
from overall import Data, task_function, set_gen

n_cuts = 3
n_settings = n_cuts ** 8


def master(ws,ds):
    print(f'I am the master! I have {ws} workers')
    print(f'Nsig = {ds.nsig}, Nbkg = {ds.nbckg}, Ntot = {ds.nevents}')

    settings = set_gen(ds, n_cuts, n_settings)
    accuracy = np.zeros(n_settings)

    #timer start
    start_time = time.time()

    # init stacks
    avail_tasks = list(range(n_settings)) # stack with idx of tasks
    avail_ws = list(range(1, comm.Get_size()))  # stack of avail workers
    task_ws = {}    # stack of which task sent to which worker
    count_ws = [0] * comm.Get_size()    # count of tasks solved per worker

    state = MPI.Status()
    while len(avail_ws) > 0:
        tidx = avail_tasks[-1]
        comm.send(settings[tidx], dest=avail_ws[-1], tag = 11)
        task_ws[avail_ws[-1]] = tidx
        avail_tasks.pop()
        avail_ws.pop()
    
    while len(avail_tasks) > 0:
        t = comm.recv(source=MPI.ANY_SOURCE,status=state)
        w = state.Get_source()
        count_ws[w] += 1
        accuracy[task_ws[w]] = t   

        tidx = avail_tasks[-1]
        comm.send(settings[tidx], dest=w, tag=11)
        task_ws[w] = tidx
        avail_tasks.pop()

    while len(avail_ws) < ws:
        t = comm.recv(source=MPI.ANY_SOURCE,status=state, tag = 11)
        w = state.Get_source()  
        count_ws[w] += 1
        accuracy[task_ws[w]] = t      

        comm.send(0, dest=w, tag=10)
        avail_ws.append(w)
        
    for w in range(1, ws + 1):
        print(f"Worker {w} solved {count_ws[w]} tasks")
    
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


def worker(rank, ds):
    state = MPI.Status()

    while True:
        sett = comm.recv(source=0, tag = MPI.ANY_TAG, status = state)
        if state.Get_tag() == 10:
            break
        acc = task_function(sett, ds)
        comm.send(acc, dest=0, tag = 11)
        

# main:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ws = comm.Get_size() - 1
Filename = 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
ds = Data(Filename)

if rank == 0:
    mpi_size = comm.Get_size()
    master(ws, ds)
else:
    worker(rank, ds)
