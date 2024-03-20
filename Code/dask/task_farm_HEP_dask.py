import sys
import numpy as np
import time
import dask.array as da
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd

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
            

def task_function(setting):
    data_less_than_setting = ds.data < setting[np.newaxis, :]
    pred = np.all(data_less_than_setting, axis=1)
    accuracy = np.sum(pred == ds.signal) / ds.nevents
    return accuracy


def set_gen(ds, n_cuts, n_settings):
    ranges = ds.means_sig + (np.arange(n_cuts) * (ds.means_bckg[:,np.newaxis] - 
                                                  ds.means_sig[:,np.newaxis]) / n_cuts).T

    settings = np.zeros((n_settings, 8)) # 8 is inner loop, that is good right?
    div = n_cuts ** np.arange(8)
    k = np.arange(n_settings)
    idx = (k[:,np.newaxis]) // div % n_cuts
    i = np.arange(8)
    settings = ranges[idx, i]

    return settings


def master():
    print(f'Dask implentation')
    settings = set_gen(ds, n_cuts, n_settings)
    # settings = np.from_array(settings, chunks=chunk_shape)      
    
    #timer start
    start_time = time.time()
    
    # Can I scatter my data before submitting?    
    accuracy = [client.submit(task_function, settings[i, :]) for i in range(n_settings)]
    accuracy = accuracy.result()
    
    idx_max_accuracy = np.argmax(accuracy)
    best_accuracy_score = accuracy[idx_max_accuracy]
    best_accuracy_setting = settings[idx_max_accuracy]
    # #timer stop
    stop_time = time.time()

    print(f'Best accuracy optained: {best_accuracy_score:.6f}')
    print('Final cuts:')
    for i in range(8):
        print(ds.name[i] + f': {ds.flip[i]*best_accuracy_setting[i]:.6f}')

    print(f'Number of settings: {n_settings}')
    print(f'Elapsed time: {stop_time-start_time} seconds')


# now running the program
if __name__ == '__main__':
    Filename = 'data/mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
    ds = Data(Filename)
    cluster = LocalCluster()
    client = Client(cluster)
    chunk_shape = (1000, 8)
    master()
    print(cluster.scheduler)