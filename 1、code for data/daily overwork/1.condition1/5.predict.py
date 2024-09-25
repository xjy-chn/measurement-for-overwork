import os
import h5py
import numpy as np
import torch
import torch.nn as nn
def train_test_split():
    for year in range(2012, 2021, 1):
        annual_files = os.listdir(fr'F:\日度夜间灯光\原始数据\result\deep\surrounding_pic\5x5_winsored\{year}')
        annual_files = [fr'F:\日度夜间灯光\原始数据\result\deep\surrounding_pic\5x5_winsored\{year}' + '\\' + file for
                        file in annual_files if file[-3:] == '.h5']
        for file in annual_files:
            with h5py.File(file, 'r') as h5:
                data = h5['data'][file[-6:-3]][:].astype(np.uint16)
            y=data[:,2,2].astype(np.uint16)
            if data.shape[0]!=0:
                data[:,2,2]=65535

                with h5py.File(f'train/{year}/{os.path.basename(file)}','w') as tr:
                    tr.create_group(name='dataset')
                    tr['dataset'].create_dataset(name='data',data=X_train,compression='gzip')
                    tr['dataset'].create_dataset(name='label', data=y_train,compression='gzip')
                with h5py.File(f'test/{year}/{os.path.basename(file)}','w') as tr:
                    tr.create_group(name='dataset')
                    tr['dataset'].create_dataset(name='data',data=X_test,compression='gzip')
                    tr['dataset'].create_dataset(name='label', data=y_test,compression='gzip')
            else:
                pass
if __name__=="__main__":
    net = torch.nn.Sequential(nn.Conv2d(1, 60, kernel_size=3), nn.Sigmoid(),
                              nn.Conv2d(60, 120, kernel_size=2), nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                              nn.Linear(120, 480), nn.Sigmoid(),
                              nn.Linear(480, 240), nn.Sigmoid(),
                              nn.Linear(240, 1))
    check_point=torch.load('model9.pt')
    print(check_point.keys())
    net.load_state_dict(check_point['model_state_dict'])
    net.eval()
    for year in range(2012, 2021, 1):
        annual_files = os.listdir(fr'F:\日度夜间灯光\原始数据\result\deep\surrounding_pic\5x5_winsored\{year}')
        annual_files = [fr'F:\日度夜间灯光\原始数据\result\deep\surrounding_pic\5x5_winsored\{year}' + '\\' + file for
                        file in annual_files if file[-3:] == '.h5']
        for file in annual_files:
            with h5py.File(file, 'r') as h5:
                data = h5['data'][file[-6:-3]][:].astype(np.uint16)
            y = data[:, 2, 2].astype(np.uint16)
            if data.shape[0] != 0:
                data[:, 2, 2] = 65535
