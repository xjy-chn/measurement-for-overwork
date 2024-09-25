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
            pass


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.net = torch.nn.Sequential(nn.Conv2d(1, 10, kernel_size=3), nn.Sigmoid(),
                                       nn.Conv2d(10, 32, kernel_size=2), nn.Sigmoid(),
                                       nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                                       nn.Linear(32, 16), nn.Sigmoid(),
                                       nn.Linear(16, 8), nn.Sigmoid(),
                                       nn.Linear(8, 1))

    def forward(self, X):
        return self.net(X)


class CustomDataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 假设我们要去掉标签为0的数据
        label = self.labels[idx]
        if label == 65535:  # 如果标签为0，返回一个错误的索引
            new_idx = idx + 1 if idx < len(self) - 1 else 0
            return self.__getitem__(new_idx)
        else:
            return self.data[idx], self.labels[idx]
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
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
    loss = nn.MSELoss().to(torch.float32)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        print(len(train_iter))

        for i, (X, y) in enumerate(train_iter):
            # print(X.shape)
            # ti=time.time()
            timer.start()
            optimizer.zero_grad()
            X = X.unsqueeze(1).to(torch.float32)
            y = y.to(torch.float32)
            y = y.squeeze(-1)
            X, y = X.to(device), y.to(device)
            # print(X.shape)
            y_hat = net(X)

            l = loss(y_hat, y).to(torch.float32)
            # print(l)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            print(timer.stop())
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
            print(timer.stop())
            print(timer.sum())
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': l,
            'seed': 0
        }, "model.pt")
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
def train_ch6_f(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.MSELoss().to(torch.float32)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        print(len(train_iter))
        X,y=train_iter.next()
        i=0
        while X is not  None:
            timer.start()
            optimizer.zero_grad()
            X = X.unsqueeze(1).to(torch.float32)
            y = y.to(torch.float32)
            y = y.squeeze(-1)
            X, y = X.to(device), y.to(device)
            # print(X.shape)
            y_hat = net(X)

            l = loss(y_hat, y).to(torch.float32)
            # print(l)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            print(timer.stop())
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
            i += 1
            X,y=train_iter.next()
            print(timer.stop())
            print(timer.sum())
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': l,
            'seed': 0
        }, "model.pt")
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
if __name__ == "__main__":
    # a=torch.tensor([1,2,1],dtype=torch.int)
    # b=torch.arange(75,dtype=torch.int).reshape((3,5,5))
    # mask=a!=2
    # c=a[mask]
    # d=b[mask]
    # print(a.shape)
    # print(b.shape)
    # print(c)
    # print(d.shape)
    # time.sleep(100)
    # torch.set_default_dtype(torch.uint16)
    with h5py.File('all_data.h5','r') as f:
        step=0
        for gname, group in f.items():
            x=group['data'][:].shape[0]
            data1 = torch.empty((x,5,5),dtype=torch.uint16)
            label=torch.empty((x),dtype=torch.uint16)
            break
        # print(data1.shape)
        for gname, group in f.items():
            if step==0:
                step+=1
            else:
                st=time.time()
                data2=torch.tensor(group['data'][:])
                label2=torch.tensor(group['label'][:])
                data1=torch.concat([data1,data2],dim=0)
                label=torch.concat([label,label2],dim=0)
                step+=1
                print(time.time()-st)

        # data1=data1.reshape((len(f)*x,5,5))
        # label=label.reshape(((len(f)*x,1)))
    label=label.to(torch.int)
    data1 = data1.to(torch.int)
    # print(label.shape)
    # print(data1.shape)
    # data1=data1[label!=65535]
    # mask=label!=65535
    # label=label[label!=65535]

    # print(len(label))
    # d=CustomDataset(data1,label)
    dataset = data.TensorDataset(data1, label)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print(train_size,test_size)
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    train_loader = data.DataLoader(train_dataset, batch_size=1024, shuffle=True,num_workers=4,prefetch_factor=10)
    test_loader = data.DataLoader(test_dataset, batch_size=1024, shuffle=True,num_workers=4,prefetch_factor=10)
    # train_prefetcher = data_prefetcher(train_loader)
    # test_prefetcher = data_prefetcher(test_loader)
    net = torch.nn.Sequential(nn.Conv2d(1, 10, kernel_size=3), nn.Sigmoid(),
                              nn.Conv2d(10, 32, kernel_size=2), nn.Sigmoid(),
                              nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                              nn.Linear(32, 16), nn.Sigmoid(),
                              nn.Linear(16, 8), nn.Sigmoid(),
                              nn.Linear(8, 1))


    train_ch6(net=net, train_iter=train_loader, test_iter=test_loader, num_epochs=5, lr=0.01, device=d2l.try_gpu())
    # for data in train_loader:
    #     imgs,target=data
    #     print(imgs.shape)
    #     print(target)
    # print(img.shape)
    # print(taret)
