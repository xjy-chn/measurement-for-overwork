import time

import h5py
# import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from torch import nn
import d2l.torch as d2l


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                # print(gname,group)
                # print(group.items())
                for dname, ds in group.items():
                    # print('ds,',ds[:])
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds[:], file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    # print('ds',ds.shape)
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds[:], file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di[
                                                                                                                 'file_path'] ==
                                                                                                             removal_keys[
                                                                                                                 0] else di
                for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


class FilteredDataset(data.Dataset):
    def __init__(self, base_dataset, filter_label):
        self.base_dataset = base_dataset
        self.filter_label = filter_label

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        data, label = self.base_dataset[index]
        if label != self.filter_label:
            return data, label
        else:
            # 如果是要去掉的标签，则返回索引超出范围的错误
            raise IndexError("Index out of range")


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.net = torch.nn.Sequential(nn.Conv2d(1, 10, kernel_size=3), nn.Sigmoid(),
                                       # nn.Conv2d(10, 32, kernel_size=2), nn.Sigmoid(),
                                       # nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                                       nn.Linear(90, 16), nn.Sigmoid(),
                                       nn.Linear(16, 8), nn.Sigmoid(),
                                       nn.Linear(8, 1))

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    # torch.set_default_dtype(torch.uint16)
    h = HDF5Dataset(file_path='F:\DeepLearning\dataLoader', recursive=False, load_data=False)
    print(h.__getitem__(0))
    # print('length',len(h))
    # label_to_filter = 65535
    # filtered_dataset = FilteredDataset(h, label_to_filter)
    # h.get_data()
    # print(h.keys())

    # print('iiiiiiiiiiiiiiiii,',img.shape)
    print(len(h))
    # time.sleep(100)
    # train_size = int(0.8 * len(h))
    # test_size = len(h) - train_size
    # train_dataset, test_dataset = data.random_split(h, [train_size, test_size])
    # train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    # net = torch.nn.Sequential(nn.Conv2d(1, 10, kernel_size=3), nn.Sigmoid(),
    #                           nn.Conv2d(10, 32, kernel_size=2), nn.Sigmoid(),
    #                           nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    #                           nn.Linear(32, 16), nn.Sigmoid(),
    #                           nn.Linear(16, 8), nn.Sigmoid(),
    #                           nn.Linear(8, 1))

    # X=torch.rand(size=(1,1,5,5),dtype=torch.float32)
    def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
        """Train a model with a GPU (defined in Chapter 6).

        Defined in :numref:`sec_utils`"""

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        net.apply(init_weights)
        print('training on', device)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        loss = nn.MSELoss()
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                legend=['train loss', 'train acc', 'test acc'])
        timer, num_batches = d2l.Timer(), len(train_iter)
        for epoch in range(num_epochs):
            # Sum of training loss, sum of training accuracy, no. of examples
            metric = d2l.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                print(X.shape)
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X.float())
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (train_l, train_acc, None))
            test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')
    # train_ch6(net=net, train_iter=train_loader, test_iter=test_loader, num_epochs=1, lr=0.01, device=d2l.try_gpu())
    # for data in train_loader:
    #     imgs,target=data
    #     print(imgs.shape)
    #     print(target)
    # print(img.shape)
    # print(taret)
