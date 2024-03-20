import numpy as np
from time import time
# Dask imports
import dask.array as da
from dask.distributed import Client, LocalCluster

if __name__ == "__main__":
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
    
    cluster = LocalCluster()  # "Each node has 36 cores and 100 gb of memory"
    client = Client(cluster)
    x = da.random.random((size, size), chunks=(1000, 1000))
    y = da.exp(x).sum()
    #y.visualize()  # Requires graphics engine. Might need to run in jupyter
    y = y.compute()

    print("-- DASK --")
    print(client.dashboard_link)  # Giver et link hvor man kan se hvad der sker
    print("Result: ", y)
    print("Time: ", time() - time_i)
    # Print scheduler info
    print("Scheduler info:\t", cluster.scheduler)