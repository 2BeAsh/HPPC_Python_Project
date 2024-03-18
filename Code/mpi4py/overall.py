import numpy as np
import pandas as pd

class Data:
    def __init__(self):
        filename = 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'
        self.name = ["averageInteractionsPerCrossing", "p_Rhad","p_Rhad1",
                     "p_TRTTrackOccupancy", "p_topoetcone40", "p_eTileGap3Cluster",
                     "p_phiModCalo", "p_etaModCalo"]
        file = pd.read_csv(filename, skiprows = 1)
        file = file.to_numpy()
        # file =  np.loadtxt(filename,delimiter=',',skiprows=1)

        self.Nvtxreco = file[:,2]
        self.p_nTracks = file[:,3]
        data_index = [1, 4, 5, 6, 7, 8, 9, 10, 11]
        self.data = file[:,data_index]
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
    #pred = np.ones(ds.nevents, dtype=bool)
    data_less_than_setting = ds.data < setting[np.newaxis, :]
    pred = np.all(data_less_than_setting, axis=1)
    
    #for i in range(8):
    #    pred = pred & (ds.data[:, i] < setting[i]) # what does this inequality mean?
    accuracy = np.sum(pred == ds.signal) / ds.nevents
    return accuracy