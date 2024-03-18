# mpiexec --mca oob_tcp_if_include lo -np 2 python3 task_farm_HEP.py
# code adapted by ida

import sys
sys.path.append("/home/jovyan/erda_mount/__dag_config__/python3")
from mpi4py import MPI
import numpy as np
import time

n_cuts = 3
n_settings = n_cuts ** 8


class Data:
    def __init__(self):
        filename = 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
        self.name = ["averageInteractionsPerCrossing", "p_Rhad","p_Rhad1",
                     "p_TRTTrackOccupancy", "p_topoetcone40", "p_eTileGap3Cluster",
                     "p_phiModCalo", "p_etaModCalo"]

        self.Nvtxreco = np.loadtxt(filename,delimiter=',',skiprows=1,usecols=2)
        self.p_nTracks = np.loadtxt(filename,delimiter=',',skiprows=1,usecols=3)
        data_index = [1, 4, 5, 6, 7, 8, 9, 10, 11]
        self.data = np.loadtxt(filename,delimiter=',',skiprows=1,usecols=data_index)
        self.signal = self.data[:,-1] == 2
        self.data = self.data[:,:-1]
        self.means_sig = np.mean(self.data[self.signal],axis=0)
        self.means_bckg = np.mean(self.data[~self.signal],axis=0)


        self.flip = np.ones(len(self.means_sig))
        for i in range(8):
            if self.means_bckg[i] < self.means_sig[i]:
                self.flip[i] = -1
                self.data[:,i] *= -1
                self.means_sig[i] *= -1
                self.means_bckg[i] *= -1
        self.nevents = len(self.data)
        self.nsig = np.sum(self.signal)
        self.nbckg = self.nevents - self.nsig

    def get(self):
        return self.data

def task_function(setting, ds):
    pred = np.ones(ds.nevents, dtype=bool)
    for i in range(8):
        pred = pred & (ds.data[:,i] < setting[i]) # what does this inequality mean?
    accuracy = np.sum(pred == ds.signal) / ds.nevents
    return accuracy

def master(ws,ds):
    print(f'I am the master! I have {ws} workers')
    print(f'Nsig = {ds.nsig}, Nbkg = {ds.nbckg}, Ntot = {ds.nevents}')

    ranges = np.zeros((n_cuts,8))
    # loop over different event channels and set up cuts

    # for i in range(8):
    for j in range(n_cuts):
        ranges[j,:] = ds.means_sig[:] + j * (ds.means_bckg[:] - ds.means_sig[:]) / n_cuts
            
    # generate list of all permutation of the cuts for each channel
    settings = np.zeros((n_settings,8))

    for k in range(n_settings):
        div = 1
        set = np.zeros(8)

        for i in range(8):
            idx = (k // div) % n_cuts
            set[i] = ranges[idx,i]
            div *= n_cuts

        settings[k,:] = set

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

ds = Data()

if rank == 0:
    mpi_size = comm.Get_size()
    master(ws, ds)
else:
    worker(rank, ds)
