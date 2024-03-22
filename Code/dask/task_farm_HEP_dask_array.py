import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from dask.distributed import LocalCluster
import time 
import sys
# sys.path.append("/home/jovyan/erda_mount/__dag_config__/python3")

class Data:
    def __init__(self, Filename='mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'):
        self.name = ["averageInteractionsPerCrossing", "p_Rhad","p_Rhad1",
                     "p_TRTTrackOccupancy", "p_topoetcone40", "p_eTileGap3Cluster",
                     "p_phiModCalo", "p_etaModCalo"]
        file = dd.read_csv(Filename, skiprows=1)  # using Dask DataFrame to read the file
        file = file.to_dask_array(lengths=True)

        self.Nvtxreco = file[:, 2]
        self.p_nTracks = file[:, 3]
        data_index = [1, 4, 5, 6, 7, 8, 9, 10, 11]
        self.data = file[:, data_index]
        self.signal = self.data[:, -1] == 2
        self.data = self.data[:, :-1]
        self.means_sig = da.mean(self.data[self.signal], axis=0)
        self.means_bckg = da.mean(self.data[~self.signal], axis=0)
        
        # Old 
        # file = file.to_numpy()        
        # self.Nvtxreco = file[:,2]
        # self.p_nTracks = file[:,3]
        # data_index = [1, 4, 5, 6, 7, 8, 9, 10, 11]
        # self.data = file[:,data_index]
        # self.signal = self.data[:,-1] == 2
        # self.data = self.data[:,:-1]
        # self.means_sig = np.mean(self.data[self.signal],axis=0)
        # self.means_bckg = np.mean(self.data[~self.signal],axis=0)


        self.flip = da.ones(len(self.means_sig))
        mask = self.means_bckg < self.means_sig
        self.flip[mask] = -1
        self.data *= self.flip
        self.means_sig = self.means_sig * self.flip
        self.means_bckg = self.means_bckg * self.flip
        self.nevents = len(self.data)
        self.nsig = da.sum(self.signal)
        self.nbckg = self.nevents - self.nsig


    
    def get(self):
        return self.data


def task_function(setting, ds, bunch_size = 1):
    if bunch_size == 1:
        if setting.ndim == 2:
            setting = setting[0,:]
        data_less_than_setting = ds.data < setting[np.newaxis, :]
        pred = np.all(data_less_than_setting, axis=1)
        accuracy = np.sum(pred == ds.signal) / ds.nevents
        return accuracy
    else:
        accuracy = np.zeros(bunch_size)
        for i in range(bunch_size):
            set = setting[i,:]
            data_less_than_setting = ds.data < set[np.newaxis, :]
            pred = np.all(data_less_than_setting, axis=1)
            accuracy[i] = np.sum(pred == ds.signal) / ds.nevents
        return accuracy


def set_gen(ds, n_cuts, n_settings):
    ranges = ds.means_sig + (np.arange(n_cuts) * (ds.means_bckg[:, np.newaxis] - 
                                                  ds.means_sig[:, np.newaxis]) / n_cuts).T

    settings = np.zeros((n_settings, 8)) # 8 is inner loop, that is good right?
    div = n_cuts**np.arange(8)
    k = np.arange(n_settings)
    idx = (k[:,np.newaxis]) // div % n_cuts
    i = np.arange(8)
    settings = ranges[idx, i]

    return settings


def master(ds):
    print(f'Nsig = {ds.nsig}, Nbkg = {ds.nbckg}, Ntot = {ds.nevents}')
    settings = set_gen(ds, n_cuts, n_settings)
    
    # loop over different event channels and set up cuts
    accuracy = np.zeros(n_settings)

    #timer start
    start_time = time.time()

    for i in range(n_settings):
        accuracy[i] = task_function(settings[i,:], ds)
    
    #timer stop
    stop_time = time.time()
        
    best_accuracy_score = np.max(accuracy)
    best_accuracy_setting = settings[np.argmax(accuracy)]

    print(f'Best accuracy optained: {best_accuracy_score:.6f}')
    print('Final cuts:')
    for i in range(8):
        print(ds.name[i] + f': {ds.flip[i]*best_accuracy_setting[i]:.6f}')

    print(f'Number of settings: {n_settings}')
    print(f'Elapsed time: {stop_time-start_time} seconds')


if __name__ == "__main__":
    # Setup Dask client
    cluster = LocalCluster()  # "Each node has 36 cores and 100 gb of memory"
    client = cluster.get_client()
    print(client.dashboard_link)  # Giver et link hvor man kan se hvad der sker
    n_cuts = 3
    n_settings = n_cuts ** 8

    Filename = 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
    ds = Data(Filename)

    master(ds)
