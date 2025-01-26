import h5py
import numpy as np
import cupy as cp
from osgeo import gdal,ogr
import os
import time
import numpy.ma as ma
import tqdm
import pandas as pd


def cal_annual_bound():
    for year in range(2012,2021):
        year_st=time.time()
        excluded_values=[65535]
        days=[day for day in os.listdir(fr'F:\ntl\raw\{year}') if day[-4:]!='.csv']
        with h5py.File(fr'./{year}firms.h5','r') as f:
            x_axis=f['data']['x'][:]
            y_axis = f['data']['y'][:]
        print(len(x_axis))
        array=-cp.ones((len(days),len(x_axis)),dtype=np.uint16)
        for i in tqdm.tqdm(range(len(days))):
            st = time.time()
            national=-cp.ones((12000, 16800), dtype=cp.uint16)
            files=os.listdir(f'./{year}/{days[i]}')
            if len(files)>0:
                for file in files:
                    block=file[17:23]
                    h=int(block[1:3])
                    v=int(block[4:6])
                    with h5py.File(f'./{year}/{days[i]}/{file}','r') as f:
                        data=cp.array(f['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['Gap_Filled_DNB_BRDF-Corrected_NTL'][:],dtype=cp.uint16)
                    national[2400*(v-3):2400*(v-2),2400*(h-25):2400*(h-24)]=data
            national=national[x_axis,y_axis]
            array[i,:]=national
        array_sum=cp.sum(cp.log(cp.where(array==65535,0,array)+1))
        array_mean=array_sum/(array.shape[0]*array.shape[1]-cp.count_nonzero(array==65535))
        mask=np.isin(array.get(),excluded_values)
        std = ma.std(ma.array(np.log(array.get()+1), mask=mask))
        print(array_mean,std,np.count_nonzero(array!=65535))

if __name__=="__main__":
    data=pd.read_csv('./interval.csv')
    cal_annual_bound()



