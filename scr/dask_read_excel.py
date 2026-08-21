import pandas as pd
import dask.dataframe as dd
import dask

def dask_read_excel(file_path):
   
   lazy_file = dask.delayed(pd.read_excel)(file_path)
   df = dd.from_delayed(lazy_file)
   
   return df