from dask.distributed import Client, LocalCluster
def dask_cluster():
    # Create a local cluster with multiple workers
    cluster = LocalCluster(
        n_workers=4,              # number of worker processes
        threads_per_worker=2,     # threads per worker
        memory_limit="4GB",       # memory cap per worker
        dashboard_address=":8787" # web UI for monitoring
    )
    client = Client(cluster)
    print(client)  # shows cluster info
    return client