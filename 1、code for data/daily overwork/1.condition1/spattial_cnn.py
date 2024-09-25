import os
import time
from torch.nn import functional as F
import torch
import torch.nn as nn
import h5py
from d2l import torch as d2l
import numpy as np
from torch.utils import data
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
import numpy.ma as ma

def train_ch61(net, train_iter, loss, optimizer, metric_train,metric_test,
               device, l_train,l_test, timer,test_iter):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""


    print('training on', device)
    net.to(device)
    net.train()
    for i, (X, y) in enumerate(tqdm(train_iter)):
        timer.start()
        optimizer.zero_grad()
        X = X.to(torch.float32)
        y = y.to(torch.float32)
        X, y = X.to(device), y.to(device)
        X = X.unsqueeze(1)
        y_hat = net(X).squeeze()
        l_train = loss(y_hat, y).to(torch.float32)
        l_train.backward()
        optimizer.step()
        # print("训练误差：",l_train)
        # print("y",y)
        # print("y_hat",y_hat)
        with torch.no_grad():
            metric_train.add(l_train * X.shape[0], X.shape[0])
            train_l = metric_train[0] / metric_train[1]
    with torch.no_grad():
        net.eval()
        for i, (X, y) in enumerate(tqdm(test_iter)):
            timer.start()
            optimizer.zero_grad()
            X = X.to(torch.float32)
            y = y.to(torch.float32)
            X, y = X.to(device), y.to(device)
            X = X.unsqueeze(1)
            y_hat = net(X).squeeze()
            l_test = loss(y_hat, y).to(torch.float32)
            metric_test.add(l_test * X.shape[0], X.shape[0])
       
        # print('-----training_loss:', train_l)
    return metric_train,metric_test, device, optimizer, l_train,l_test, timer
def train_ch6(net, train_iter, loss, optimizer, metric, device, l, timer, label_mean=None, label_std=None, x_mean=None, x_std=None):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""

    print('training on', device)
    net.to(device)
    net.train()

    for i, (X, y) in enumerate(tqdm(train_iter)):
        # print(X.shape)
        # ti=time.time()
        timer.start()
        optimizer.zero_grad()
        X = X.to(torch.float32)
        y = y.to(torch.float32)
        # X = (X - x_mean) / x_std
        # y = (y - label_mean) / label_std
        # X = torch.where(X > 100, torch.full_like(X, 65535), X)
        X, y = X.to(device), y.to(device)
        # print(X.shape)
        X = X.unsqueeze(1)
        y_hat = net(X).squeeze()
        l = loss(y_hat, y).to(torch.float32)
        # print(l)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], X.shape[0])
        train_l = metric[0] / metric[1]
        # print('-----training_loss:', train_l)
    return metric, device, optimizer, l, timer


def train_ch62(net, train_iter, lr, device, metric):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.MSELoss().to(torch.float32)
    timer, num_batches = d2l.Timer(), len(train_iter)
    net.train()
    for i, (X, y) in enumerate(train_iter):
        # print(X.shape)
        # ti=time.time()
        timer.start()
        optimizer.zero_grad()
        X = X.unsqueeze(1).to(torch.float32)
        y = y.to(torch.float32)
        y = y.squeeze(-1)
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        print(y)
        print(y_hat)
        l = loss(y_hat, y).to(torch.float32)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
    return metric


def init_seed(seed=2019, reproducibility=True) -> None:
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
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
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
if __name__ == "__main__":
    seed=100
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
    num_of_files = 55
    epochs = 10
    init_seed(seed)
    if not os.path.isfile('train_loss.csv'):
        with open('train_loss.csv', 'w', encoding='utf-8') as f:
            f.write('epoch,num,train_loss\n')
    # net = torch.nn.Sequential(nn.Conv2d(1, 16, kernel_size=3), nn.Sigmoid(),
    #                           nn.Conv2d(16, 48, kernel_size=2), nn.Sigmoid(),
    #                           nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    #                           nn.Linear(48, 120), nn.Sigmoid(),
    #                           # nn.Linear(480, 240), nn.Sigmoid(),
    #                           nn.Linear(120, 1))
    # net2 = torch.nn.Sequential(nn.Conv2d(1, 60, kernel_size=3), nn.Sigmoid(),
    #                           nn.Conv2d(60, 120, kernel_size=2), nn.Sigmoid(),
    #                           nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    #                           nn.Linear(120, 480), nn.Sigmoid(),
    #                           nn.Linear(480, 240), nn.Sigmoid(),
    #                           nn.Linear(240, 1))
    timer = d2l.Timer()
    lr = 0.001
    optimizer = torch.optim.Adam(net3.parameters(), lr=lr, eps=0.006)
    loss = nn.MSELoss().to(torch.float32)
    metric_train = d2l.Accumulator(2)
    metric_test = d2l.Accumulator(2)
    device = d2l.try_gpu()
    l_train = 0
    l_test=0
    net3.apply(init_weights)
    for i in range(0, epochs):
        train_dataset_fps = os.listdir(fr'data/{i}/train')
        test_dataset_fps = os.listdir(fr'data/{i}/test')
        train_dataset_fps = [fr'data/{i}/train' + '//' + fp for fp in train_dataset_fps]
        test_dataset_fps = [fr'data/{i}/test' + '//' + fp for fp in test_dataset_fps]
        metric_train = d2l.Accumulator(2)
        metric_test = d2l.Accumulator(2)
        for d in range(num_of_files):
            with h5py.File(train_dataset_fps[d], 'r') as f:
                train_data = torch.Tensor(f['data']['X'][:]).to(torch.uint16)
                train_label = torch.Tensor(f['data']['label'][:]).to(torch.uint16)
                # label_mean = np.mean(label.numpy())
                # label_std = np.std(label.numpy())
                # array = data.numpy()
                # mask = np.isin(array, [65535])
                # mean = ma.mean(ma.array(array, mask=mask))
                # std = ma.std(ma.array(array, mask=mask))
                # array = None
            with h5py.File(test_dataset_fps[d], 'r') as f:
                test_data = torch.Tensor(f['data']['X'][:]).to(torch.uint16)
                test_label = torch.Tensor(f['data']['label'][:]).to(torch.uint16)
            train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
            test_dataset= torch.utils.data.TensorDataset(test_data, test_label)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
            if i == 0 & d == 0:
                metric_train,metric_test, device, optimizer, l_train,l_test, timer = train_ch61(net3, train_loader, loss=loss,
                                                                 optimizer=optimizer, device=device,
                                                                 metric_train=metric_train,
                                                                 metric_test=metric_test, l_train=None,
                                                                 l_test=None, timer=timer, test_iter=test_loader)
                torch.save({
                    'epoch': i,
                    'part':d,
                    'model_state_dict': net3.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': l_train,
                    'seed': seed
                }, f"model{i}part{d}.pt")
            else:
                metric_train, metric_test, device, optimizer, l_train, l_test, timer = train_ch61(net3, train_loader,
                                                                                                  loss=loss,
                                                                                                  optimizer=optimizer,
                                                                                                  device=device,
                                                                                                  metric_train=metric_train,
                                                                                                  metric_test=metric_test,
                                                                                                  l_train=l_train,
                                                                                                  l_test=l_test,
                                                                                                  timer=timer,
                                                                                                  test_iter=test_loader)
                torch.save({
                    'epoch': i,
                    'part':d,
                    'model_state_dict': net3.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': l_train,
                    'seed': 100,
                    "loss_value_train":metric_train[0] / metric_train[1],
                    "loss_value_test":metric_test[0] / metric_test[1]
                }, f"model{i}part{d}.pt")
            train_l = metric_train[0] / metric_train[1]
            print('-----training_loss:', train_l)
            print('-----epoch:', i + 1)
            test_l = metric_test[0] / metric_test[1]
            print('-----test_loss:', test_l)
            print('-----epoch:', i + 1)
            if (d + 1) % (num_of_files // 5) == 0 or i == num_of_files - 1:
                with open('./train_loss.csv', 'a', encoding='utf-8') as f:
                    f.write(f'{i},{train_l},' + f'{(d + 1) % num_of_files}' + '\n')
        torch.save({
            'epoch': i,
            'model_state_dict': net3.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': l_train,
            'seed': seed,
            "loss_value_train":metric_train[0] / metric_train[1],
            "loss_value_test":metric_test[0] / metric_test[1]
        }, f"model{i}.pt")
        # animator.add(i + 1, (None, None))
        print(f'loss {train_l:.3f}')
