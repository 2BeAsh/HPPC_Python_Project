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
    cluster = LocalCluster()
    client = Client(cluster)
    x = da.random.random((size, size), chunks=(1000, 1000))
    y = da.exp(x).sum().compute()

    print("-- DASK --")
    print("Result: ", y)
    print("Time: ", time() - time_i)
    # Print scheduler info
    print(cluster.scheduler)