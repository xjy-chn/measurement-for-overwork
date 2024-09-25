import json
import time

import numpy as np
from osgeo import gdal
import pandas as pd

if __name__=="__main__":
    year=2012

    missing=dict()
    for year in range(2012,2021):
        missing[f'{year}']=[]
    for year in range(2012,2021):
        dataset = gdal.Open(fr'F:\日度夜间灯光\结果\{year}\缺失_{year}_裁切.tif')
        data = dataset.ReadAsArray()
        c1 = data >= 0
        c2 = data <= 30
        c3 = data <= 60
        c4 = data <= 100
        c5 = data > 100
        c6 = data > 30
        c7 = data > 60
        missing[f'{year}'].append(np.count_nonzero(c1&c2)/np.count_nonzero(c1))
        missing[f'{year}'].append(np.count_nonzero(c6 & c3)/np.count_nonzero(c1))
        missing[f'{year}'].append(np.count_nonzero(c7 & c4)/np.count_nonzero(c1))
        missing[f'{year}'].append(np.count_nonzero(c5)/np.count_nonzero(c1))
    data=pd.DataFrame(missing).T
    data.columns=['leq30','leq60','leq60','teg100']
    data.to_excel('./result/原始数据缺失情况.xlsx')