import sys
import numpy as np
import time
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd


class Data:
    def __init__(self, filename):
        self.name = ["averageInteractionsPerCrossing", "p_Rhad","p_Rhad1",
                     "p_TRTTrackOccupancy", "p_topoetcone40", "p_eTileGap3Cluster",
                     "p_phiModCalo", "p_etaModCalo"]
        df = dd.read_csv(filename, assume_missing=True)  # assume_missing=True for better performance. Should not change results

        # Not all columns are relevant
        data_index = [1, 4, 5, 6, 7, 8, 9, 10, 11]
        self.data = df.iloc[:, data_index]
        
        # Extract signal and remove it from the dataframe
        self.signal = self.data.iloc[:, -1] == 2
        self.data = self.data.iloc[:, :-1]
                
        # Find means
        means_sig = self.data[self.signal].mean(axis=0)
        means_bckg = self.data[~self.signal].mean(axis=0)
        
        # Convert means to array
        means_sig_array = means_sig.values
        means_bckg_array = means_bckg.values
        
        # Flip calculations
        self.flip_array = da.where(means_bckg_array < means_sig_array, -1, 1).compute()
        
        # Multiply flip onto data and means
        self.data = self.data.values * self.flip_array
        self.means_sig = means_sig_array.compute() * self.flip_array
        self.means_bckg = means_bckg_array.compute() * self.flip_array
        
        # The remaining attributes        
        self.nevents = len(self.signal)
            

def task_function(setting, data, signal, nevents):
    data_less_than_setting = data < setting[np.newaxis, :]
    pred = da.all(data_less_than_setting, axis=1)
    accuracy = da.sum(pred == signal) / nevents
    return accuracy.compute()


def set_gen(means_sig, means_bckg, n_cuts, n_settings):
    # Convert to dask array
    means_sig_array = da.from_array(means_sig)
    means_bckg_array = da.from_array(means_bckg)
    
    # Compute ranges
    ranges = means_sig + (da.arange(n_cuts) * (means_bckg_array[:,np.newaxis] - 
                                                  means_sig_array[:,np.newaxis]) / n_cuts).T
    
    # Dask has not implemented fancy indenting, so workaround using flatten
    flat_ranges = da.ravel(ranges)
    
    # Generate indices
    div = n_cuts ** da.arange(8)
    k = da.arange(n_settings)
    idx = (k[:, np.newaxis]) // div % n_cuts
    
    # Flat idx so matches flat_ranges
    flat_idx = da.ravel(idx)
    
    # Use flattened idx on flattened ranges and reshape back
    settings = flat_ranges[flat_idx].reshape((n_settings, 8))
    
    return settings


def scatter_data(ds, client):
    data_future = client.scatter(ds.data)
    signal_future = client.scatter(ds.signal)
    nevents_future = client.scatter(ds.nevents)
    return data_future, signal_future, nevents_future


def master(filename, n_cuts, n_settings, client):
    # Timer start
    start_time = time.time()

    # Read data
    ds = Data(filename)
    
    # Compute settings
    settings = set_gen(ds.means_sig, ds.means_bckg, n_cuts, n_settings)
    print("Got settings")
    # Scatter data to workers
    #data_future, signal_future, nevents_future = scatter_data(ds, client)
    
    #accuracy_futures = [client.submit(task_function, setting, data_future, signal_future, nevents_future) for setting in settings]
    #accuracy_futures = [client.submit(task_function, setting, ds.data, ds.signal, ds.nevents) for setting in settings]
    #accuracy = client.gather(accuracy_futures)
    accuracy = da.apply_along_axis(task_function, axis=1, arr=settings, data=ds.data, signal=ds.signal, nevents=ds.nevents)
    print("Got accuracies")
    
    # Find best accuracy
    idx_max_accuracy = da.argmax(accuracy)
    best_accuracy_score = accuracy[idx_max_accuracy].compute()
    best_accuracy_setting = settings[idx_max_accuracy].compute()
    
    # Timer stop
    stop_time = time.time()

    print(f'Best accuracy optained: {best_accuracy_score:.6f}')
    print('Final cuts:')
    for i in range(8):
        print(ds.name[i] + f': {ds.flip_array[i]*best_accuracy_setting[i]:.6f}')

    print(f'Number of settings: {n_settings}')
    print(f'Elapsed time: {stop_time-start_time} seconds')


# Run the program
if __name__ == '__main__':
    filename = 'data/mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
    cluster = LocalCluster()
    client = Client(cluster)
    #chunk_shape = (1000, 8)
    n_cuts = 3
    n_settings = n_cuts ** 8
    master(filename, n_cuts, n_settings, client)
    print(cluster.scheduler)