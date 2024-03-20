import numpy as np
from time import time
# Dask imports
import dask.array as da
from dask.distributed import Client
from dask_jobqueue import PBSCluster

size = 10_000
# NUMPY
time_np = time()
x_np = np.random.random((size, size))
y_np = np.exp(x_np).sum()
print("-- NUMPY --")
print("Result: ", y_np)
print("Time: ", time() - time_np)

# DASK
time_i = time()   
cluster = PBSCluster(cores=36, memory="12GB")  # "Each node has 36 cores and 12 gb of memory"
cluster.scale(n=2)  # n workers. Antal computere. Each with the specifications given in PBSCluster
client = Client(cluster)
x = da.random.random((size, size), chunks=(1000, 1000))
y = da.exp(x).sum()
y = y.compute()

print("-- DASK --")
print("Result: ", y)
print("Time: ", time() - time_i)
print("Scheduler info:\t", cluster.scheduler)  # Print scheduler info