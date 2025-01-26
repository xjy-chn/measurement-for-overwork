import os
import time

import numpy as np
import json
import h5py
import numpy.ma as ma
import cupy as cp
import pandas as pd

def search_day_dirs(year):
    dirs = os.listdir(fr'F:\ntl\raw\{year}')
    dirs = [fr'F:\ntl\raw\{year}' + '/' + dir for dir in dirs]
    dirs = [dir for dir in dirs if os.path.isdir(dir)]
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
    # seperate holidays festivals and workdays
    festivals, holidays, weekends, works, all = [], [], [], [], []
    with open(fr'./calendar/{data_year}.json', 'r', encoding='utf-8') as file:
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
        dataset = file['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['Gap_Filled_DNB_BRDF-Corrected_NTL']
        data_array = cp.array(dataset)
    return data_array


def read_median_h5(fp, block):
    with h5py.File(fp, 'r') as file:
        dataset = file['data'][block]
        data_array = cp.array(dataset)
    return data_array





def collect_block_files(year, type, annual_type):
    for block in blocks:
        block_files = []
        for day in type:
            key = str(day['dayOfYear']).zfill(3)
            for file in daily_files[fr'F:\ntl\raw\{year}/' + key]:
                if block in file:
                    block_files.append(fr'F:\ntl\raw\{year}/' + file)
                    break
        annual_type[block] = block_files
    return annual_type




def save_move_self(data, x,y,year, day,description, r,bound=False):
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
            f['imformation'].create_dataset(name='description', data=description)
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
            f['imformation'].create_dataset(name='description', data=description)
            f['imformation'].create_dataset(name='radius', data=r)

def extract_surrounding(left, right,r,winsor=False):
    if winsor:
        end = 0

        for key, value in daily_files.items():
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
                view = cp.lib.stride_tricks.as_strided(padded_ntl, shape=(12000, 16800, r, r),
                                                       strides=((16800+(r-1)) * 2, 2, (16800+(r-1)) * 2, 2))

                firmnum=cp.zeros((12000,16800),dtype=cp.uint16)
                with h5py.File(f'./{year}firms_position.h5', "r") as f:
                    xindex=f['data']['x'][:]
                    yindex = f['data']['y'][:]
                    firmnum[xindex,yindex]= 1
                x,y=cp.where(firmnum>0)
                view2=view[x,y,:,:]
                x=cp.array(x,dtype=cp.uint16)
                y=cp.array(y,dtype=cp.uint16)
                print(x.shape,y.shape)

                save_move_self(data=view2.get().astype(np.uint16),x=x.get(),y=y.get(), year=year, day=day,
                                        description=f"data for the {day} in {year},running time:{time.time() - st}",
                                        r=r,bound=True)
    else:
        end = 0
        for key, value in daily_files.items():
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
                    data = None
                padded_ntl = cp.pad(ntl, ((((r - 1) // 2), ((r - 1) // 2)), (((r - 1) // 2), ((r - 1) // 2))),
                                    'constant', constant_values=(65535, 65535))
                print(padded_ntl.shape)
                view = cp.lib.stride_tricks.as_strided(padded_ntl, shape=(12000, 16800, r, r),
                                                       strides=(
                                                           (16800 + (r - 1)) * 2, 2, (16800 + (r - 1)) * 2, 2))
                print(view[7002, 12259, :, :])
                print(ntl[7000:7005, 12257:12262])
                firmnum = cp.zeros((12000, 16800), dtype=cp.uint16)
                with h5py.File(f'./{year}firms_position.h5', "r") as f:
                    xindex = f['data']['x'][:]
                    yindex = f['data']['y'][:]
                    firmnum[xindex, yindex] = 1
                x, y = cp.where(firmnum > 0)
                view2 = view[x, y, :, :]
                view = None
                x = cp.array(x, dtype=cp.uint16)
                y = cp.array(y, dtype=cp.uint16)
                print(x.shape, y.shape)
                save_move_self(data=view2.get().astype(np.uint16), x=x.get(), y=y.get(), year=year, day=day,
                               description=f"data for the {day} in {year},running time:{time.time() - st}",
                               r=r, bound=False)
    view2 = None
    x=None
    y=None
    end += time.time() - st



def cal_bound(mean,std):
    l=np.exp(mean-3*std)-1
    r=np.exp(mean+3*std)-1
    return l,r

if __name__ == "__main__":
    values_to_exclude = [65535]
    blocks = construct_blocks()
    extreme = pd.read_csv('./interval.csv')
    for year in range(2012, 2021):
        l, r = cal_bound(extreme.loc[year-2012,'mean'],extreme.loc[year-2012,'std'])
        if year != 2022:
            day_dirs = search_day_dirs(year)
            files = [search_h5_files(path) for path in day_dirs]
            daily_files = dict(zip(day_dirs, files))
            print(daily_files)
            print(daily_files.keys())
            extract_surrounding(l,r,r=5)
