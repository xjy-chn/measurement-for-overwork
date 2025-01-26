import time

import h5py

import pandas as pd

import cupy as cp


if __name__=="__main__":
    radius=5
    for year in range(2012,2021):
        st=time.time()
        with h5py.File(f"./result/annual_overwork/deep2/{year}_dummy.h5", 'r') as f:
            overwork = cp.array(f['data']['dummy'][:], dtype=cp.uint8)
            x_axis = f['data']['x'][:]
            y_axis = f['data']['y'][:]
            print(x_axis,y_axis)
            print(len(x_axis),cp.max(x_axis))
            # print(cp.max(overwork))
        with h5py.File(f'./{year}firms_position.h5', "r") as f:
            firmnum = cp.array(f['data'][f'{year}'][:], dtype=cp.uint16)
            firmnum = firmnum[x_axis,y_axis]
            print("max num",cp.max(firmnum))
        dots=pd.DataFrame([overwork.get(),firmnum.get()]).T
        dots.columns=['overwork','firms']
        dots.to_csv(f"./result/variables/deep2/{radius}x{radius}_winsor/{year}dots.csv",index=False)
        print(f"cost{time.time()-st} sec")