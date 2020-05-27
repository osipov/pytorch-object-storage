import os
import math
import tempfile

import fsspec
import pandas as pd
import numpy as np
import torch as pt

from torch.utils.data import IterableDataset

class ObjectStorageDataset(IterableDataset):

  def __init__(self, glob, batch_size = None, iterations = None,  cache_dir = None, storage_options = None, fits_in_node_memory = True, fits_in_cluster_memory = True, worker = 0, replicas = 1):

    self.glob = glob

    #set the platform-specific temporary directory
    cache_dir = cache_dir if cache_dir else tempfile.gettempdir()

    #find out the protocol of the glob, e.g. s3, gs, hdfs, etc
    protocol, _ = fsspec.core.split_protocol(glob)

    #use anonymous connection unless specified otherwise
    storage_options = storage_options if storage_options else {'anon': True}

    #setup a caching filesystem
    fs = fsspec.filesystem("filecache",
                            target_protocol=protocol,
                            target_options=storage_options,
                            cache_storage=cache_dir)

    #get the object paths matching the glob
    self.objs = fs.glob(glob)

    if fits_in_node_memory:

      #read the contents of the objects into a dataframes list
      dfs = list()
      for obj in self.objs:
        with fs.open(obj) as file:
          dfs.append(pd.read_csv(file))

      #concatenate the dataframes (if any) together
      self.df = pd.concat(dfs) if len(dfs) > 0 else pd.DataFrame()

    elif fits_in_cluster_memory:

      assert replicas > 0 and type(replicas) is int, "The number of workers must be a positive integer"
      assert worker > -1 and worker < replicas, "The worker must be in the range [0, replicas)"

      self.worker = worker
      self.replicas = replicas

      #TODO: fix the DRY issue
      self.objects_per_worker = int(math.ceil(len(self.objs) / self.replicas))

      dfs = list()
      for obj in self.objs[worker * self.objects_per_worker : (worker + 1) * self.objects_per_worker]:
        with fs.open(obj) as file:
          dfs.append(pd.read_csv(file))
      self.df = pd.concat(dfs) if len(dfs) > 0 else pd.DataFrame()

    else:
      self.df = pd.DataFrame()

    self.dataset_size = len(self.df)
    self.batch_size = batch_size if batch_size else self.dataset_size
    self.iterations = iterations if iterations else float('nan')

  def __iter__(self):

    idx = 0

    while self.iterations:
      df_range = np.arange(idx, idx + self.batch_size) % self.dataset_size

      yield pt.tensor(self.df.iloc[ df_range].values)

      idx = (idx + self.batch_size) % self.dataset_size
      self.iterations -= 1
