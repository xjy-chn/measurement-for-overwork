import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import d2l.torch as d2l

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
if __name__=="__main__":
    device = d2l.try_gpu()
    print(device)
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.Sigmoid(),
                       nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net3 = nn.Sequential(b1, b2, b3, b4, b5,
                         nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(), nn.Linear(512, 1))
    check_point = torch.load('model3part10.pt')
    print(check_point.keys())
    net3.load_state_dict(check_point['model_state_dict'])
    net3.eval()
    loss = nn.MSELoss().to(torch.float32)
    for year in range(2012, 2021, 1):
        if not os.path.exists(f'./predict/{year}'):
        annual_files = os.listdir(fr'F:\ntl\raw\{year}')
        annual_files = [fr'F:\ntl\raw\{year}' + '\\' + file for
                        file in annual_files if file[-3:] == '.h5']
        for file in tqdm(annual_files):
            with h5py.File(file, 'r') as h5:
                data = h5['data'][file[-6:-3]][:].astype(np.uint16)
                # print(h5['data'].keys())
                x_axis = h5['data']['x'][:].astype(np.uint16)
                y_axis = h5['data']['y'][:].astype(np.uint16)
            label = data[:, 2, 2].astype(np.uint16)
            array = data
            if data.shape[0] != 0:
                data[:, 2, 2] = 65535
            net3.to(device)
            X = torch.tensor(data).to(torch.float32).unsqueeze(1)
            y = torch.tensor(label).to(torch.float32)
            X.to(device)
            y.to(device)
            net3.to(device)
            net3.eval()
            dataset = torch.utils.data.TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=X.shape[0], shuffle=False)
            with torch.no_grad():
                for i, (d, l) in enumerate(loader):
                    print(d.shape)
                    d, l = d.to(device), l.to(device)
                    y_hat2 = net3(d)
                    with h5py.File(f'./predict/{year}/' + os.path.basename(file), 'w') as f:
                        f.create_group('data')
                        f['data'].create_dataset(name='predict', data=y_hat2.squeeze(-1).cpu(), compression="gzip")
                        f['data'].create_dataset(name='label', data=label, compression="gzip")
                        f['data'].create_dataset(name='x', data=x_axis, compression="gzip")
                        f['data'].create_dataset(name='y', data=y_axis, compression="gzip")
                        f['data'].create_dataset(name='dataset', data=data, compression="gzip")