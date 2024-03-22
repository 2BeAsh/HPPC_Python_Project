import sys
sys.path.append("/home/jovyan/erda_mount/__dag_config__/python3")
# from mpi4py import MPI
import numpy as np
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
from dask_mpi import initialize
import time
import logging

logger = logging.getLogger("__name__")
logger.setLevel(logging.ERROR)

n_cuts = 3
n_settings = n_cuts ** 8

class Data:
    def __init__(self,Filename='mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'):
        self.name = ["averageInteractionsPerCrossing", "p_Rhad","p_Rhad1",
                     "p_TRTTrackOccupancy", "p_topoetcone40", "p_eTileGap3Cluster",
                     "p_phiModCalo", "p_etaModCalo"]
        file = pd.read_csv(Filename, skiprows = 1) # because pandas is faster than numpy
        file = file.to_numpy()

        self.Nvtxreco = file[:,2]
        self.p_nTracks = file[:,3]
        data_index = [1, 4, 5, 6, 7, 8, 9, 10, 11]
        self.data = file[:,data_index]
        self.signal = self.data[:,-1] == 2
        self.data = self.data[:,:-1]
        self.means_sig = np.mean(self.data[self.signal],axis=0)
        self.means_bckg = np.mean(self.data[~self.signal],axis=0)


        self.flip = np.ones(len(self.means_sig))
        mask = self.means_bckg < self.means_sig
        self.flip[mask] = -1
        self.data *= self.flip
        self.means_sig = self.means_sig * self.flip
        self.means_bckg = self.means_bckg * self.flip
        self.nevents = len(self.data)
        self.nsig = np.sum(self.signal)
        self.nbckg = self.nevents - self.nsig

    def get(self):
        return self.data

def task_function(setting):

    data_less_than_setting = ds.data < setting[np.newaxis, :]
    pred = np.all(data_less_than_setting, axis=1)
    accuracy = np.sum(pred == ds.signal) / ds.nevents
    return accuracy


def set_gen(ds, n_cuts, n_settings):
    ranges = ds.means_sig + (np.arange(n_cuts) * (ds.means_bckg[:,np.newaxis] - 
                                                  ds.means_sig[:,np.newaxis]) / n_cuts).T

    settings = np.zeros((n_settings,8)) # 8 is inner loop, that is good right?
    div = n_cuts**np.arange(8)
    k = np.arange(n_settings)
    idx = (k[:,np.newaxis]) // div % n_cuts
    i = np.arange(8)
    settings = ranges[idx,i]

    return settings

def master():

    settings = set_gen(ds, n_cuts, n_settings)
    cs = 100

    #timer start
    start_time = time.time()

    settings = da.from_array(settings).rechunk(cs, 8)
    accuracy = da.apply_along_axis(task_function, axis = 1, arr = settings)
        
    best_accuracy_score = da.max(accuracy).compute()
    best_accuracy_setting = settings[da.argmax(accuracy)].compute()

    #timer stop
    stop_time = time.time()
    
    ws = len(client.scheduler_info()['workers'])
    with open("dask_arrays_" + str(ws) + ".txt", "w") as file:
        file.write(f'Using dask arrays with chunk size {cs},8\n')
        file.write(f'Best accuracy optained: {best_accuracy_score:.6f} \n')
        file.write('Final cuts:\n')
        for i in range(8):
            file.write(ds.name[i] + f': {ds.flip[i]*best_accuracy_setting[i]:.6f}\n')
    
        file.write(f'Number of settings: {n_settings}\n')
        file.write(f'Elapsed time: {stop_time-start_time} seconds\n')
        file.write(f"Workers: {len(client.scheduler_info()['workers'])}\n")


# Run the program
if __name__ == '__main__':
    filename = 'data/mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
    initialize()
    with Client() as client:
        ds = Data(filename)
        n_cuts = 3
        n_settings = n_cuts ** 8
        master()
