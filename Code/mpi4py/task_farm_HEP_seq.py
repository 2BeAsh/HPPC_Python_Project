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






def master(mpi_size,ds):
    print(f'I am the master! I have {mpi_size-1} workers')
    print(f'Nsig = {ds.nsig}, Nbkg = {ds.nbckg}, Ntot = {ds.nevents}')

    ranges = np.zeros((n_cuts,8))
    # loop over different event channels and set up cuts

    # for i in range(8):
    for j in range(n_cuts):
        ranges[j,:] = ds.means_sig[:] + j * (ds.means_bckg[:] - ds.means_sig[:]) / n_cuts
    # for i in range(8):
    #     for j in range(n_cuts):
    #         ranges[j,i] = ds.means_sig[i] + j * (ds.means_bckg[i] - ds.means_sig[i]) / n_cuts
    # # vectorize above loops
            
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

    for i in range(n_settings):
        accuracy[i] = task_function(settings[i],ds)
    
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
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ds = Data()

if rank == 0:
    # comm.send(data, dest=1, tag=11)
    mpi_size = comm.Get_size()
    master(mpi_size,ds)
else:
    # data = comm.recv(source=0, tag=11)
    worker(rank,ds)
