import os
import time

import numpy as np
import json
import h5py
import numpy.ma as ma
import cupy as cp
import pandas as pd

def search_day_dirs(year):
    dirs = os.listdir(f'./{year}')
    dirs = [f'./{year}' + '/' + dir for dir in dirs]
    dirs = [dir for dir in dirs if os.path.isdir(dir)]
    # print(dirs)
    return dirs


def search_h5_files(path):
    return os.listdir(path)


def construct_blocks():
    blocks = []
    delete_blocks = ['h25v07', 'h26v07', 'h27v07', 'h30v07', 'h31v07']
    for h in range(25, 32):
        for v in range(3, 8):
            block = "h{}v{}".format(h, str(v).zfill(2))
            blocks.append(block)
    # print(blocks)
    for block in delete_blocks:
        blocks.remove(block)
    return blocks


def get_days(data_year):
    # 节假日、假日、周末、工作日
    festivals, holidays, weekends, works, all = [], [], [], [], []
    with open(f'./万年历/{data_year}.json', 'r', encoding='utf-8') as file:
        content = json.load(file)
        data = content['data']
        for month in data:
            for day in month['days']:
                if day['type'] == 2:
                    festivals.append(day)
                    holidays.append(day)
                    all.append(day)
                elif day['type'] == 1:
                    holidays.append(day)
                    weekends.append(day)
                    all.append(day)
                elif day['type'] == 0:
                    works.append(day)
                    all.append(day)
    return festivals, holidays, weekends, works, all


def read_raw_h5(fp):
    with h5py.File(fp, 'r') as file:
        # 读取数据集到 NumPy 数组
        dataset = file['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['Gap_Filled_DNB_BRDF-Corrected_NTL']
        data_array = cp.array(dataset)
        # 无效值替换为空
        # nan_indices = cp.where(data_array == 65535)
        # data_array[nan_indices] = cp.nan
    return data_array


def read_median_h5(fp, block):
    with h5py.File(fp, 'r') as file:
        # 读取数据集到 NumPy 数组
        dataset = file['data'][block]
        data_array = cp.array(dataset)
        # 无效值替换为空
        # nan_indices = cp.where(data_array == 65535)
        # data_array[nan_indices] = cp.nan
    return data_array


def save(data, block, year, description):
    filename = block
    # print(filename)
    if not os.path.exists(f'./result/{year}'):
        os.makedirs(f'./result/{year}')
    with h5py.File(f'./result/{year}' + '/' + filename + '.h5', "w") as f:
        f.create_group('imformation')
        f.create_group('data')
        f['data'].create_dataset(name=block, data=data)
        f['imformation'].create_dataset(name='基本信息', data=description)


def save_overwork(data, block, year, description):
    filename = block
    # print(filename)
    if not os.path.exists(f'./result/{year}'):
        os.makedirs(f'./result/{year}')
    with h5py.File(f'./result/{year}' + '/' + filename + '.h5', "w") as f:
        f.create_group('imformation')
        f.create_group('data')
        f['data'].create_dataset(name=block, data=data)
        f['imformation'].create_dataset(name='基本信息', data=description)


def collect_block_files(year, type, annual_type):
    for block in blocks:
        block_files = []
        for day in type:
            key = str(day['dayOfYear']).zfill(3)
            for file in daily_files[f'./{year}/' + key]:
                if block in file:
                    block_files.append(f'./{year}/{key}/' + file)
                    break
        annual_type[block] = block_files
    return annual_type


def cal_median(year, annual_type):
    # 处理数据
    for key, value in annual_type.items():
        daynum = len(value)
        array = cp.zeros((daynum, 2400, 2400), dtype=cp.uint16)
        for i in range(daynum):
            array[i] = read_raw_h5(value[i])

        mask = cp.isin(array, values_to_exclude)
        # time.sleep(100)
        median_value = ma.median(ma.array(array, mask=mask), axis=0)
        print(median_value.shape)
        description = f"{year}年{key}地区的节假日中位数"
        save(data=median_value, block=key, year=year, description=description)


def save_surrounding_pic(data, x,y,year, day, description, r,bound=True):
    # print(filename)
    if bound:
        if not os.path.exists(f'./result/deep/surrounding_pic/{r}x{r}_winsored/{year}'):
            os.makedirs(f'./result/deep/surrounding_pic/{r}x{r}_winsored/{year}')
        with h5py.File(f'./result/deep/surrounding_pic/{r}x{r}_winsored/{year}/{day}.h5', "w") as f:
            f.create_group('imformation')
            f.create_group('data')
            f['data'].create_dataset(name=day, data=data,compression="gzip")
            f['data'].create_dataset(name='x', data=x, compression="gzip")
            f['data'].create_dataset(name='y', data=y, compression="gzip")
            f['imformation'].create_dataset(name='基本信息', data=description)
            f['imformation'].create_dataset(name='radius', data=r)
    else:
        if not os.path.exists(f'./result/deep/surrounding_pic/{year}'):
            os.makedirs(f'./result/deep/surrounding_pic/{year}')
        with h5py.File(f'./result/deep/surrounding_pic/{year}/{day}.h5', "w") as f:
            f.create_group('imformation')
            f.create_group('data')
            f['data'].create_dataset(name=day, data=data, compression="gzip")
            f['data'].create_dataset(name='x', data=x, compression="gzip")
            f['data'].create_dataset(name='y', data=y, compression="gzip")
            f['imformation'].create_dataset(name='基本信息', data=description)
            f['imformation'].create_dataset(name='radius', data=r)
def save_move(data,firm, x,y,year, day, direction,description, r,bound=True):
    # print(filename)
    if bound:
        if not os.path.exists(f'./result/deep/surrounding_pic/{r}x{r}_firmwinsored/{year}/{direction}'):
            os.makedirs(f'./result/deep/surrounding_pic/{r}x{r}_firmwinsored/{year}/{direction}')
        with h5py.File(f'./result/deep/surrounding_pic/{r}x{r}_firmwinsored/{year}/{direction}/{day}.h5', "w") as f:
            f.create_group('imformation')
            f.create_group('data')
            f['data'].create_dataset(name=day, data=data,compression="gzip")
            f['data'].create_dataset(name='firm', data=firm, compression="gzip")
            f['data'].create_dataset(name='x', data=x, compression="gzip")
            f['data'].create_dataset(name='y', data=y, compression="gzip")
            f['imformation'].create_dataset(name='基本信息', data=description)
            f['imformation'].create_dataset(name='radius', data=r)
    else:
        if not os.path.exists(f'./result/deep/surrounding_pic/{year}/{direction}'):
            os.makedirs(f'./result/deep/surrounding_pic/{year}/{direction}')
        with h5py.File(f'./result/deep/surrounding_pic/{year}/{direction}/{day}.h5', "w") as f:
            f.create_group('imformation')
            f.create_group('data')
            f['data'].create_dataset(name=day, data=data, compression="gzip")
            f['data'].create_dataset(name='firm', data=firm, compression="gzip")
            f['data'].create_dataset(name='x', data=x, compression="gzip")
            f['data'].create_dataset(name='y', data=y, compression="gzip")
            f['imformation'].create_dataset(name='基本信息', data=description)
            f['imformation'].create_dataset(name='radius', data=r)

def save_move_self(data, x,y,year, day,description, r,bound=True):
    # print(filename)
    if bound:
        if not os.path.exists(f'./result/deep/surrounding_pic/{r}x{r}_firmwinsored/{year}'):
            os.makedirs(f'./result/deep/surrounding_pic/{r}x{r}_firmwinsored/{year}')
        with h5py.File(f'./result/deep/surrounding_pic/{r}x{r}_firmwinsored/{year}/{day}.h5', "w") as f:
            f.create_group('imformation')
            f.create_group('data')
            f['data'].create_dataset(name=day, data=data,compression="gzip")
            # f['data'].create_dataset(name='firm', data=firm, compression="gzip")
            f['data'].create_dataset(name='x', data=x, compression="gzip")
            f['data'].create_dataset(name='y', data=y, compression="gzip")
            f['imformation'].create_dataset(name='基本信息', data=description)
            f['imformation'].create_dataset(name='radius', data=r)
    else:
        if not os.path.exists(f'./result/deep/surrounding_pic/{year}'):
            os.makedirs(f'./result/deep/surrounding_pic/{year}')
        with h5py.File(f'./result/deep/surrounding_pic/{year}/{day}.h5', "w") as f:
            f.create_group('imformation')
            f.create_group('data')
            f['data'].create_dataset(name=day, data=data, compression="gzip")
            # f['data'].create_dataset(name='firm', data=firm, compression="gzip")
            f['data'].create_dataset(name='x', data=x, compression="gzip")
            f['data'].create_dataset(name='y', data=y, compression="gzip")
            f['imformation'].create_dataset(name='基本信息', data=description)
            f['imformation'].create_dataset(name='radius', data=r)

def cal_surrounding_median_dummy(left, right,r):
    end = 0
    # pinned_mempool = cp.get_default_pinned_memory_pool()

    for key, value in daily_files.items():
        # pinned_mempool.free_all_blocks()
        day = key[-3:]
        year = key[2:6]
        st=time.time()
        if not os.path.isfile(f'./result/deep/surrounding_pic/{r}x{r}_firmwinsored/{year}/{day}.h5'):
            st = time.time()
            ntl = -cp.ones((5 * 2400, 7 * 2400), dtype=cp.uint16)
            for block in value:
                h = int(block[18:20])
                v = int(block[21:23])
                data = read_raw_h5(key + '/' + block)
                ntl[2400 * (v - 3):2400 * (v - 2), 2400 * (h - 25):2400 * (h - 24)] = data
                data=None
            c1 = ntl < left
            c2 = ntl > right
            c3 = ntl != 65535
            ntl = cp.where(c1 & c3, 65535, ntl)
            ntl = cp.where(c2 & c3, 65535, ntl)
            padded_ntl = cp.pad(ntl, ((((r-1)//2), ((r-1)//2)), (((r-1)//2), ((r-1)//2))), 'constant', constant_values=(65535, 65535))
            print(padded_ntl.shape)
            view = cp.lib.stride_tricks.as_strided(padded_ntl, shape=(12000, 16800, r, r),
                                                   strides=((16800+(r-1)) * 2, 2, (16800+(r-1)) * 2, 2))
            with h5py.File(fr'./全部企业栅格索引/{year}firms.h5','r') as f:
                x=f['data']['x'][:]
                y=f['data']['y'][:]
                print(len(x),x)
            view2=view[x,y,:,:]
            view=None
            x=cp.array(x,dtype=cp.uint16)
            y=cp.array(y,dtype=cp.uint16)
            print(x.shape,y.shape)
            # time.sleep(100)
            save_move_self(data=view2.get().astype(np.uint16),x=x.get(),y=y.get(), year=year, day=day,
                                    description=f"这是全国{year}年度{day}天的数据,用时{time.time() - st}",
                                    r=r)
            view2=None
            x=None
            y=None
            print(f"这是全国{year}年度{day}天的数据,用时{time.time() - st}")
        print(f"这是全国{year}年度{day}天的数据保存完毕")
        end += time.time() - st
        print("用时：", end)

def move_surrounding_up(left, right,r):
    end = 0
    # pinned_mempool = cp.get_default_pinned_memory_pool()

    for key, value in daily_files.items():
        # pinned_mempool.free_all_blocks()
        day = key[-3:]
        year = key[2:6]
        if not os.path.isfile(f'./result/deep/surrounding_pic/{year}/{day}.h5'):
            st = time.time()
            ntl = -cp.ones((5 * 2400, 7 * 2400), dtype=cp.uint16)
            for block in value:
                h = int(block[18:20])
                v = int(block[21:23])
                data = read_raw_h5(key + '/' + block)
                ntl[2400 * (v - 3):2400 * (v - 2), 2400 * (h - 25):2400 * (h - 24)] = data
                data=None
            c1 = ntl < left
            c2 = ntl > right
            c3 = ntl != 65535
            ntl = cp.where(c1 & c3, 65535, ntl)
            ntl = cp.where(c2 & c3, 65535, ntl)
            padded_ntl = cp.pad(ntl, ((((r - 1) // 2), ((r - 1) // 2)), (((r - 1) // 2), ((r - 1) // 2))), 'constant',
                                constant_values=(65535, 65535))
            print(padded_ntl.shape)
            view = cp.lib.stride_tricks.as_strided(padded_ntl, shape=(12000, 16800, r, r),
                                                   strides=((16800 + (r - 1)) * 2, 2, (16800 + (r - 1)) * 2, 2))
            with h5py.File(f'./年度企业存量栅格/{year}firms_position.h5', "r") as f:
                firmnum = cp.array(f['data'][f'{year}'][:], dtype=cp.uint16)
            padded_firmnum = cp.pad(ntl, ((((r-1)//2), ((r-1)//2)), (((r-1)//2), ((r-1)//2))), 'constant', constant_values=(0, 0))
            view_firm = cp.lib.stride_tricks.as_strided(padded_firmnum, shape=(12000, 16800, r, r),
                                                        strides=((16800 + r - 1) * 2, 2, (16800 + r - 1) * 2, 2))
            x,y=cp.where(firmnum>0)
            print(x)
            x,y=cp.array(x-2,dtype=cp.uint16),cp.array(y,dtype=cp.uint16)
            print(x)
            valid=cp.where(x>=0)
            print('有效长度',x.shape,valid[0].shape)
            # print(list(valid.get()))
            valid_x=x[valid[0]]
            valid_y=y[valid[0]]
            firms=firmnum[valid_x,valid_y]
            print(firms.shape)
            x=cp.where(firms==0)
            x_valid=valid_x[x]
            y_valid = valid_y[x]
            print(x_valid.shape,y_valid.shape)
            # time.sleep(100)
            view2=view[x_valid,y_valid,:,:]
            view_firm2=view_firm[x_valid,y_valid,:,:]
            view=None
            x_valid=cp.array(x_valid,dtype=cp.uint16)
            y_valid=cp.array(y_valid,dtype=cp.uint16)
            # print(x.shape,y.shape)
            # time.sleep(100)
            save_move(data=view2.get().astype(np.uint16),firm=view_firm2.get(),x=x_valid.get(),y=y_valid.get(), year=year,direction='up', day=day,
                                    description=f"这是全国{year}年度{day}天的数据,用时{time.time() - st}",
                                    r=r)
            view2=None
            x=None
            y=None
            print(f"这是全国{year}年度{day}天的数据,用时{time.time() - st}")
        print(f"这是全国{year}年度{day}天的数据保存完毕")
        end += time.time() - st
        print("用时：", end)
def move_surrounding_down(left, right,r):
    end = 0
    # pinned_mempool = cp.get_default_pinned_memory_pool()

    for key, value in daily_files.items():
        # pinned_mempool.free_all_blocks()
        day = key[-3:]
        year = key[2:6]
        if not os.path.isfile(f'./result/deep/surrounding_pic/{year}/{day}.h5'):
            st = time.time()
            ntl = -cp.ones((5 * 2400, 7 * 2400), dtype=cp.uint16)
            for block in value:
                h = int(block[18:20])
                v = int(block[21:23])
                data = read_raw_h5(key + '/' + block)
                ntl[2400 * (v - 3):2400 * (v - 2), 2400 * (h - 25):2400 * (h - 24)] = data
                data=None
            c1 = ntl < left
            c2 = ntl > right
            c3 = ntl != 65535
            ntl = cp.where(c1 & c3, 65535, ntl)
            ntl = cp.where(c2 & c3, 65535, ntl)
            padded_ntl = cp.pad(ntl, ((((r-1)//2), ((r-1)//2)), (((r-1)//2), ((r-1)//2))), 'constant', constant_values=(65535, 65535))
            print(padded_ntl.shape)
            view = cp.lib.stride_tricks.as_strided(padded_ntl, shape=(12000, 16800, r, r),
                                                   strides=((16800+(r-1)) * 2, 2, (16800+(r-1)) * 2, 2))
            with h5py.File(f'./年度企业存量栅格/{year}firms_position.h5', "r") as f:
                firmnum = cp.array(f['data'][f'{year}'][:], dtype=cp.uint16)
            padded_firmnum = cp.pad(ntl, ((((r - 1) // 2), ((r - 1) // 2)), (((r - 1) // 2), ((r - 1) // 2))),
                                    'constant', constant_values=(0, 0))
            view_firm = cp.lib.stride_tricks.as_strided(padded_firmnum, shape=(12000, 16800, r, r),
                                                        strides=((16800 + r - 1) * 2, 2, (16800 + r - 1) * 2, 2))
            x,y=cp.where(firmnum>0)
            print(x)
            x,y=cp.array(x+2,dtype=cp.uint16),cp.array(y,dtype=cp.uint16)
            print(x)
            valid=cp.where(x<12000)
            print('有效长度',x.shape,valid[0].shape)
            # print(list(valid.get()))
            valid_x=x[valid[0]]
            valid_y=y[valid[0]]
            firms=firmnum[valid_x,valid_y]
            print(firms.shape)
            x=cp.where(firms==0)
            x_valid=valid_x[x]
            y_valid = valid_y[x]
            print(x_valid.shape,y_valid.shape)
            # time.sleep(100)
            view2=view[x_valid,y_valid,:,:]
            view_firm2=view_firm[x_valid,y_valid,:,:]
            view=None
            x_valid=cp.array(x_valid,dtype=cp.uint16)
            y_valid=cp.array(y_valid,dtype=cp.uint16)
            # print(x.shape,y.shape)
            # time.sleep(100)
            save_move(data=view2.get().astype(np.uint16),firm=view_firm2.get(),x=x_valid.get(),y=y_valid.get(), year=year,direction='down', day=day,
                                    description=f"这是全国{year}年度{day}天的数据,用时{time.time() - st}",
                                    r=r)
            view2=None
            x=None
            y=None
            print(f"这是全国{year}年度{day}天的数据,用时{time.time() - st}")
        print(f"这是全国{year}年度{day}天的数据保存完毕")
        end += time.time() - st
        print("用时：", end)
def move_surrounding_left(left, right,r):
    end = 0
    # pinned_mempool = cp.get_default_pinned_memory_pool()

    for key, value in daily_files.items():
        # pinned_mempool.free_all_blocks()
        day = key[-3:]
        year = key[2:6]
        if not os.path.isfile(f'./result/deep/surrounding_pic/{year}/{day}.h5'):
            st = time.time()
            ntl = -cp.ones((5 * 2400, 7 * 2400), dtype=cp.uint16)
            for block in value:
                h = int(block[18:20])
                v = int(block[21:23])
                data = read_raw_h5(key + '/' + block)
                ntl[2400 * (v - 3):2400 * (v - 2), 2400 * (h - 25):2400 * (h - 24)] = data
                data=None
            c1 = ntl < left
            c2 = ntl > right
            c3 = ntl != 65535
            ntl = cp.where(c1 & c3, 65535, ntl)
            ntl = cp.where(c2 & c3, 65535, ntl)
            padded_ntl = cp.pad(ntl, ((((r - 1) // 2), ((r - 1) // 2)), (((r - 1) // 2), ((r - 1) // 2))), 'constant',
                                constant_values=(65535, 65535))
            print(padded_ntl.shape)
            view = cp.lib.stride_tricks.as_strided(padded_ntl, shape=(12000, 16800, r, r),
                                                   strides=((16800 + (r - 1)) * 2, 2, (16800 + (r - 1)) * 2, 2))
            with h5py.File(f'./年度企业存量栅格/{year}firms_position.h5', "r") as f:
                firmnum = cp.array(f['data'][f'{year}'][:], dtype=cp.uint16)
            padded_firmnum = cp.pad(ntl, ((((r - 1) // 2), ((r - 1) // 2)), (((r - 1) // 2), ((r - 1) // 2))),
                                    'constant', constant_values=(0, 0))
            view_firm = cp.lib.stride_tricks.as_strided(padded_firmnum, shape=(12000, 16800, r, r),
                                                        strides=((16800 + r - 1) * 2, 2, (16800 + r - 1) * 2, 2))
            x,y=cp.where(firmnum>0)
            print(x)
            x,y=cp.array(x,dtype=cp.uint16),cp.array(y-2,dtype=cp.uint16)
            print(x)
            valid=cp.where(y>=0)
            print('有效长度',x.shape,valid[0].shape)
            # print(list(valid.get()))
            valid_x=x[valid[0]]
            valid_y=y[valid[0]]
            firms=firmnum[valid_x,valid_y]
            print(firms.shape)
            x=cp.where(firms==0)
            x_valid=valid_x[x]
            y_valid = valid_y[x]
            print(x_valid.shape,y_valid.shape)
            # time.sleep(100)
            view2=view[x_valid,y_valid,:,:]
            view_firm2=view_firm[x_valid,y_valid,:,:]
            view=None
            x_valid=cp.array(x_valid,dtype=cp.uint16)
            y_valid=cp.array(y_valid,dtype=cp.uint16)
            # print(x.shape,y.shape)
            # time.sleep(100)
            save_move(data=view2.get().astype(np.uint16),firm=view_firm2.get(),x=x_valid.get(),y=y_valid.get(), year=year,direction='left', day=day,
                                    description=f"这是全国{year}年度{day}天的数据,用时{time.time() - st}",
                                    r=r)
            view2=None
            x=None
            y=None
            print(f"这是全国{year}年度{day}天的数据,用时{time.time() - st}")
        print(f"这是全国{year}年度{day}天的数据保存完毕")
        end += time.time() - st
        print("用时：", end)
def move_surrounding_right(left, right,r):
    end = 0
    # pinned_mempool = cp.get_default_pinned_memory_pool()

    for key, value in daily_files.items():
        # pinned_mempool.free_all_blocks()
        day = key[-3:]
        year = key[2:6]
        if not os.path.isfile(f'./result/deep/surrounding_pic/{year}/{day}.h5'):
            st = time.time()
            ntl = -cp.ones((5 * 2400, 7 * 2400), dtype=cp.uint16)
            for block in value:
                h = int(block[18:20])
                v = int(block[21:23])
                data = read_raw_h5(key + '/' + block)
                ntl[2400 * (v - 3):2400 * (v - 2), 2400 * (h - 25):2400 * (h - 24)] = data
                data=None
            c1 = ntl < left
            c2 = ntl > right
            c3 = ntl != 65535
            ntl = cp.where(c1 & c3, 65535, ntl)
            ntl = cp.where(c2 & c3, 65535, ntl)
            padded_ntl = cp.pad(ntl, ((((r-1)//2), ((r-1)//2)), (((r-1)//2), ((r-1)//2))), 'constant', constant_values=(65535, 65535))
            print(padded_ntl.shape)
            view = cp.lib.stride_tricks.as_strided(padded_ntl, shape=(12000, 16800, r, r),
                                                   strides=((16800+(r-1)) * 2, 2, (16800+(r-1)) * 2, 2))
            with h5py.File(f'./年度企业存量栅格/{year}firms_position.h5', "r") as f:
                firmnum = cp.array(f['data'][f'{year}'][:], dtype=cp.uint16)
            padded_firmnum = cp.pad(ntl, ((((r - 1) // 2), ((r - 1) // 2)), (((r - 1) // 2), ((r - 1) // 2))),
                                    'constant', constant_values=(0, 0))
            view_firm = cp.lib.stride_tricks.as_strided(padded_firmnum, shape=(12000, 16800, r, r),
                                                        strides=((16800 + r - 1) * 2, 2, (16800 + r - 1) * 2, 2))
            x,y=cp.where(firmnum>0)
            print(x)
            x,y=cp.array(x,dtype=cp.uint16),cp.array(y+2,dtype=cp.uint16)
            print(x)
            valid=cp.where(y<16800)
            print('有效长度',x.shape,valid[0].shape)
            # print(list(valid.get()))
            valid_x=x[valid[0]]
            valid_y=y[valid[0]]
            firms=firmnum[valid_x,valid_y]
            print(firms.shape)
            x=cp.where(firms==0)
            x_valid=valid_x[x]
            y_valid = valid_y[x]
            print(x_valid.shape,y_valid.shape)
            # time.sleep(100)
            view2=view[x_valid,y_valid,:,:]
            view_firm2=view_firm[x_valid,y_valid,:,:]
            view=None
            x_valid=cp.array(x_valid,dtype=cp.uint16)
            y_valid=cp.array(y_valid,dtype=cp.uint16)
            # print(x.shape,y.shape)
            # time.sleep(100)
            save_move(data=view2.get().astype(np.uint16),firm=view_firm2.get(),x=x_valid.get(),y=y_valid.get(), year=year,direction='right', day=day,
                                    description=f"这是全国{year}年度{day}天的数据,用时{time.time() - st}",
                                    r=r)
            view2=None
            x=None
            y=None
            print(f"这是全国{year}年度{day}天的数据,用时{time.time() - st}")
        print(f"这是全国{year}年度{day}天的数据保存完毕")
        end += time.time() - st
        print("用时：", end)
def cal_bound(mean,std):
    l=np.exp(mean-3*std)-1
    r=np.exp(mean+3*std)-1
    return l,r

if __name__ == "__main__":
    # time.sleep(3*3600)
    values_to_exclude = [65535]
    blocks = construct_blocks()
    # for year in range(2012,2025):
    #     if year not in [2012,2022,2024]:
    extreme=pd.read_csv('./计算正态区间/interval.csv')

    for year in range(2012, 2021):
        l, r = cal_bound(extreme.loc[year-2012,'mean'],extreme.loc[year-2012,'std'])
        print(l,r)


        left, right = cp.exp(l) - 1, cp.exp(r) - 1
        # print(left,right)
        # time.sleep(100)
        if year != 2022:
            day_dirs = search_day_dirs(year)
            files = [search_h5_files(path) for path in day_dirs]
            daily_files = dict(zip(day_dirs, files))
            print(daily_files)
            print(daily_files.keys())
            cal_surrounding_median_dummy(left,right,r=5)
            # move_surrounding_up(left, right, r=5)
            # move_surrounding_down(left,right,r=5)
            # move_surrounding_left(left, right, r=5)
            # move_surrounding_right(left, right, r=5)
