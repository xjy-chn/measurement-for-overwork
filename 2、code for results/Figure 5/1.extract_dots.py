import time

import h5py
import os
import numpy as np
import pandas as pd
# import cupy as cp
from osgeo import ogr,gdal
import cupy as cp
def read_raw_h5(fp):
    with h5py.File(fp, 'r') as file:
        # 读取数据集到 NumPy 数组
        dataset = file['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['Gap_Filled_DNB_BRDF-Corrected_NTL']
        data_array = cp.array(dataset)
        # 无效值替换为空
        # nan_indices = cp.where(data_array == 65535)
        # data_array[nan_indices] = cp.nan
    return data_array

#
# fp='./2012/022/VNP46A2.A2012022.h31v06.001.2020038183150.h5'
# fp2='./2013/002/VNP46A2.A2013002.h31v06.001.2020129220638.h5'
# data=read_raw_h5(fp)
# print(cp.min(data))
# print(cp.count_nonzero(data==0))
# print(data)

# with h5py.File(r'F:\日度夜间灯光\原始数据\年度企业存量栅格\2012firms_position.h5','r') as f:
#     data=np.array(f['data']['2012'][:])
# print(data)
# print(np.max(data))
# x,y=np.where(data!=0)
# print(len(x))

def clip_tiff(shapefile_path, tiff_path, output_path):
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()
    tiff_dataset = gdal.Open(tiff_path)
    output_dataset = gdal.Warp(output_path, tiff_dataset, cutlineDSName=shapefile_path, cutlineLayer=layer.GetName(),
                               cropToCutline=True)
    print(tiff_dataset)
    print(shapefile)
    print(output_dataset)
    data=cp.array(output_dataset.ReadAsArray())
    print(cp.count_nonzero(cp.array(output_dataset.ReadAsArray())!=101))
    output_dataset.GetRasterBand(1).ComputeStatistics(0)
    output_dataset = None
    tiff_dataset = None
    shapefile = None
    return data
def arr_to_tiff(array, fp, data_type,nodata):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(fp, array.shape[1], array.shape[0], 1, data_type)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.GetRasterBand(1).SetNoDataValue(nodata)
    dataset.GetRasterBand(1).ComputeStatistics(0)

    shp_ds = ogr.Open(f'./国界/国.shp')
    layer = shp_ds.GetLayer()
    spatial_ref = layer.GetSpatialRef()
    # 设置输出 GeoTIFF 文件的地理参考信息
    dataset.SetProjection(spatial_ref.ExportToWkt())
    # 设置地理变换参数
    rows, cols = array.shape
    x_min, x_max, y_min, y_max = layer.GetExtent()
    pixel_width = (x_max - x_min) / cols
    pixel_height = (y_max - y_min) / rows
    dataset.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_height))

    dataset.FlushCache()
    dataset = None
    shp_ds = None

if __name__=="__main__":
    radius=3
    if not  os.path.exists(f"./result/variables/deep/{radius}x{radius}_winsor/描述性统计/工商注册栅格层面相关性和散点图/tif/CN"):
        os.makedirs(f"./result/variables/deep/{radius}x{radius}_winsor/描述性统计/工商注册栅格层面相关性和散点图/tif/CN")
    for year in range(2012,2021):
        st=time.time()
        with h5py.File(f"./result/annual_overwork/deep/{year}_dummy.h5", 'r') as f:
            overwork = cp.array(f['data']['dummy'][:], dtype=cp.uint8)
            x_axis = f['data']['x'][:]
            y_axis = f['data']['y'][:]
            print(x_axis,y_axis)
            print(len(x_axis),cp.max(x_axis))
            # print(cp.max(overwork))
        with h5py.File(f'./年度企业存量栅格/{year}firms_position.h5', "r") as f:
            firmnum = cp.array(f['data'][f'{year}'][:], dtype=cp.uint16)
            firmnum = firmnum[x_axis,y_axis]
            print(cp.max(firmnum))
        dots=pd.DataFrame([overwork.get(),firmnum.get()]).T
        dots.columns=['overwork','firms']
        dots.to_csv(f"./result/variables/deep/{radius}x{radius}_winsor/描述性统计/工商注册栅格层面相关性和散点图/{year}dots.csv",index=False)
        print(f"用时{time.time()-st}秒")