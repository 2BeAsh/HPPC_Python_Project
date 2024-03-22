import sys
import numpy as np
import time
from dask.distributed import Client
from dask_mpi import initialize
import numpy as np
import pandas as pd

n_cuts = 3
n_settings = n_cuts ** 8

class Data:
    def __init__(self, filename):
        self.name = ["averageInteractionsPerCrossing", "p_Rhad","p_Rhad1",
                     "p_TRTTrackOccupancy", "p_topoetcone40", "p_eTileGap3Cluster",
                     "p_phiModCalo", "p_etaModCalo"]
        file = pd.read_csv(filename, skiprows = 1) # because pandas is faster than numpy
        file = file.to_numpy()

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
            

def task_function(setting, data, signal, nevents):
    data_less_than_setting = data < setting[np.newaxis, :]
    pred = np.all(data_less_than_setting, axis=1)
    accuracy = np.sum(pred == signal) / nevents
    return accuracy


def set_gen(means_sig, means_bckg, n_cuts, n_settings):
    ranges = means_sig + (np.arange(n_cuts) * (means_bckg[:,np.newaxis] - 
                                                  means_sig[:,np.newaxis]) / n_cuts).T
    settings = np.zeros((n_settings, 8)) # 8 is inner loop, that is good right?
    div = n_cuts ** np.arange(8)
    k = np.arange(n_settings)
    idx = (k[:,np.newaxis]) // div % n_cuts
    i = np.arange(8)
    settings = ranges[idx, i]
    return settings


def scatter_data(ds, client):
    data_future = client.scatter(ds.data)
    signal_future = client.scatter(ds.signal)
    nevents_future = client.scatter(ds.nevents)
    return data_future, signal_future, nevents_future


def master(ds, n_cuts, n_settings, client):
    #timer start
    start_time = time.time()

    # Get settings
    settings = set_gen(ds.means_sig, ds.means_bckg, n_cuts, n_settings)
    
    # Scatter data to workers
    data_future, signal_future, nevents_future = scatter_data(ds, client)
    
    # Get futures
    accuracy_futures = [client.submit(task_function, settings[i, :], data_future, signal_future, nevents_future) for i in range(n_settings)]
    # Get Future values
    accuracy = client.gather(accuracy_futures)
    
    idx_max_accuracy = np.argmax(accuracy)
    best_accuracy_score = accuracy[idx_max_accuracy]
    best_accuracy_setting = settings[idx_max_accuracy]
    # #timer stop
    stop_time = time.time()

    ws = len(client.scheduler_info()['workers'])
    with open("futures_" + str(ws) + ".txt", "w") as file:
        file.write(f'Best accuracy optained: {best_accuracy_score:.6f} \n')
        file.write('Final cuts:\n')
        for i in range(8):
            file.write(ds.name[i] + f': {ds.flip[i]*best_accuracy_setting[i]:.6f}\n')
    
        file.write(f'Number of settings: {n_settings}\n')
        file.write(f'Elapsed time: {stop_time-start_time} seconds\n')
        file.write(f"Workers: {len(client.scheduler_info()['workers'])}\n")


# now running the program
if __name__ == '__main__':
    initialize()
    filename = 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
    ds = Data(filename)
    client = Client()
    master(ds, n_cuts, n_settings, client)
