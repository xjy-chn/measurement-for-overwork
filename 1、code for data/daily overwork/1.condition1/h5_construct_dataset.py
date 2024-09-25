import h5py
import os
import time
import numpy as np
import torch
# a=torch.arange(16).reshape((2,2,2,2))
# print(a)
# print(a.reshape(4,2,2))
# print
hdf5_files=os.listdir('../data/NTL/2012')
hdf5_files=['../data/NTL/2012'+'//'+h for h in hdf5_files if h[-3:]=='.h5']
hdf5_files=hdf5_files[0:2]
with h5py.File('example.h5', 'w') as h:
# with h5py.File('all_data.h5', 'w') as h:
    st=time.time()
    for file in hdf5_files:
        t=time.time()
        with h5py.File(file,'r') as f:
            data=f['data'][file[-6:-3]][:]
            data=data[data[:,2,2]!=65535]
            label=data[:,2,2]
            print(label)
            h.create_group(name=file[-6:-3])
            h[file[-6:-3]].create_dataset(name='data', data=data, compression='gzip')
            h[file[-6:-3]].create_dataset(name='label', data=label)
            print(f'{file[-6:-3]}天i耗时{time.time() - t}秒,已用{time.time() - st}秒')
#             # break
