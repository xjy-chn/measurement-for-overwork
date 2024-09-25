import time
import numpy as np
import os
import h5py
from tqdm import tqdm
import cupy as cp

def train_test_split(train_ratio=0.8):
    for year in tqdm(range(2012, 2021, 1)):
        annual_files = os.listdir(fr'F:\日度夜间灯光\原始数据\result\deep\surrounding_pic\5x5_firmwinsored\{year}')
        annual_files = [fr'F:\日度夜间灯光\原始数据\result\deep\surrounding_pic\5x5_firmwinsored\{year}' + '\\' + file for
                        file in annual_files if file[-3:] == '.h5']
        if not os.path.exists(f'train/{year}'):
            os.makedirs(f'train/{year}')
        if not os.path.exists(f'test/{year}'):
            os.makedirs(f'test/{year}')

        for file in annual_files:
            with h5py.File(file, 'r') as h5:
                data = h5['data'][file[-6:-3]][:].astype(np.uint16)
            y=data[:,2,2].astype(np.uint16)
            pos=np.where(y!=65535)
            y=y[pos]
            data=data[pos,:,:]
            data=data.reshape((data.shape[1],5,5))
            # print(data.shape,data.shape[0],data.shape[1])
            if data.shape[0]!=0:
                data[:,2,2]=65535

                # 随机打乱数据的索引
                shuffled_indexes = np.random.permutation(len(data))

                # 确定测试集的大小，例如20%作为测试集
                test_size = int(len(data) * (1-train_ratio))

                # 分离测试集和训练集
                test_indexes = shuffled_indexes[:test_size]
                train_indexes = shuffled_indexes[test_size:]

                # 划分数据集
                X_test = data[test_indexes,:,:]
                y_test = y[test_indexes]
                X_train = data[train_indexes,:,:]
                y_train = y[train_indexes]
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


def sampling_train():
    all_files = []
    length=[]
    for year in tqdm(range(2012,2021,1)):
        annual_files=os.listdir(fr'train\{year}')
        annual_files=[fr'train\{year}'+'\\'+file for file in annual_files if file[-3:]=='.h5']
        all_files=all_files+annual_files
    # print(all_files)
    num_of_files=len(all_files)
    num_of_datasets=num_of_files//60+1
    # print(num_of_files,num_of_datasets)
    indexes=np.arange(num_of_files)
    num_of_epochs=5
    print("开始随机抽取训练集")
    for e in tqdm(range(num_of_epochs)):
        groups = np.array_split(np.random.choice(indexes, len(indexes), replace=False), num_of_datasets)
        for i in range(len(groups)):
            dataset = np.arange(25,dtype=np.uint16).reshape((1,5,5))
            labels = np.arange(1, dtype=np.uint16)
            step=0
            for j in groups[i]:
                step+=1
                with h5py.File(all_files[j],'r') as h5:
                    data=h5['dataset']['data'][:]
                    label=h5['dataset']['label'][:]
                    dataset=np.concatenate((dataset,data),axis=0)
                    labels = np.concatenate((labels,label), axis=0)
                    # print(step)

            dataset=dataset[1:,:,:]
            labels=labels[1:]
            labels=labels.astype(np.uint16)
            dataset = dataset.astype(np.uint16)
            print('数据形状为：',dataset.shape)
            if not os.path.exists(f'./{e}/train'):
                os.makedirs(f'./{e}/train')
            with h5py.File(f'{e}/train/part{i}.h5', 'w') as f:
                f.create_group(name='data')
                f['data'].create_dataset(name='X',data=dataset,compression='gzip')
                f['data'].create_dataset(name='label', data=labels, compression='gzip')
def sampling_test():
    all_files = []
    length=[]
    for year in tqdm(range(2012,2021,1)):
        annual_files=os.listdir(fr'test\{year}')
        annual_files=[fr'test\{year}'+'\\'+file for file in annual_files if file[-3:]=='.h5']
        all_files=all_files+annual_files
    # print(all_files)
    num_of_files=len(all_files)
    num_of_datasets=num_of_files//60+1
    # print(num_of_files,num_of_datasets)
    indexes=np.arange(num_of_files)
    num_of_epochs=5
    print("开始分epoch抽取测试集")
    for e in tqdm(range(num_of_epochs)):
        groups = np.array_split(np.random.choice(indexes, len(indexes), replace=False), num_of_datasets)
        for i in range(len(groups)):
            dataset = np.arange(25,dtype=np.uint16).reshape((1,5,5))
            labels = np.arange(1, dtype=np.uint16)
            step=0
            for j in groups[i]:
                step+=1
                with h5py.File(all_files[j],'r') as h5:
                    data=h5['dataset']['data'][:]
                    label=h5['dataset']['label'][:]
                    dataset=np.concatenate((dataset,data),axis=0)
                    labels = np.concatenate((labels,label), axis=0)
                    # print(step)

            dataset=dataset[1:,:,:]
            labels=labels[1:]
            labels=labels.astype(np.uint16)
            dataset = dataset.astype(np.uint16)
            print('数据形状为：',dataset.shape)
            if not os.path.exists(f'./{e}/test'):
                os.makedirs(f'./{e}/test')
            with h5py.File(f'{e}/test/part{i}.h5', 'w') as f:
                f.create_group(name='data')
                f['data'].create_dataset(name='X',data=dataset,compression='gzip')
                f['data'].create_dataset(name='label', data=labels, compression='gzip')
if __name__=="__main__":
    np.random.seed(100)
    print("开始划分训练和测试集")

    # train_test_split()
    print("开始分割训练集")
    # sampling_train()
    print("开始分割测试集")
    sampling_test()