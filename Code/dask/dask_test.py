import numpy as np
# Dask imports
import dask.array as da
from dask.distributed import Client, LocalCluster


if __name__ == "__main__":
    # DASK
    cluster = LocalCluster()  # "Each node has 36 cores and 100 gb of memory"
    client = Client(cluster)
    print(client.dashboard_link)  # Giver et link hvor man kan se hvad der sker

    size = 10_000

    x = da.random.random((size, size), chunks=(1000, 1000))
    y = da.exp(x).sum()
    #y.visualize()  # Requires graphics engine. Might need to run in jupyter
    y = y.compute()

    print("-- DASK --")
    print("Result: ", y)
    print("Scheduler info:\t", cluster.scheduler) # Print scheduler info