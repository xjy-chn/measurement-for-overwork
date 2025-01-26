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
    for block in delete_blocks:
        blocks.remove(block)
    return blocks


def get_days(data_year):
    # seperate holidays festivals and workdays
    festivals, holidays, weekends, works, all = [], [], [], [], []
    if data_year != 2012:
        with open(f'./calendar/{data_year}.json', 'r', encoding='utf-8') as file:
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
    elif data_year == 2012:
        with open(f'./calendar/{data_year}.json', 'r', encoding='utf-8') as file:
            content = json.load(file)
            data = content['data']
            for month in data:
                for day in month['days']:
                    if day['type'] == 2 and int(day['dayOfYear'])>=19:
                        festivals.append(day)
                        holidays.append(day)
                        all.append(day)
                    elif day['type'] == 1 and int(day['dayOfYear'])>=19:
                        holidays.append(day)
                        weekends.append(day)
                        all.append(day)
                    elif day['type'] == 0 and int(day['dayOfYear'])>=19:
                        works.append(day)
                        all.append(day)
    return festivals, holidays, weekends, works, all


def read_raw_h5(fp):
    with h5py.File(fp, 'r') as file:
        dataset = file['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['Gap_Filled_DNB_BRDF-Corrected_NTL']
        data_array = cp.array(dataset)
    return data_array



def collect_block_files(year, type, annual_type):
    days = list(daily_files.keys())
    days = [day[-3:] for day in days]
    for block in blocks:
        block_files = []
        for day in type:
            key = str(day['dayOfYear']).zfill(3)
            if key in days:
                for file in daily_files[f'./{year}/' + key]:
                    if block in file:
                        block_files.append(f'./{year}/{key}/' + file)
                        break
        annual_type[block] = block_files
    return annual_type




def save_no_missing(data, year, block, description,type,bound=True):
    # print(filename)
    if bound:
        if not os.path.exists(f'./result/overwork_nomissing/winsor/deep2/{year}/{type}'):
            os.makedirs(f'./result/overwork_nomissing/winsor/deep2/{year}/{type}')
        with h5py.File(f'./result/overwork_nomissing/winsor/deep2/{year}/{type}' + '/' + block + '.h5', "w") as f:
            f.create_group('information')
            f.create_group('data')
            f['data'].create_dataset(name=block, data=data)
            f['information'].create_dataset(name='description', data=description)
    else:
        if not os.path.exists(f'./result/overwork_nomissing/{year}/{type}'):
            os.makedirs(f'./result/overwork_nomissing/{year}/{type}')
        with h5py.File(f'./result/overwork_nomissing/{year}/{type}' + '/' + block + '.h5', "w") as f:
            f.create_group('information')
            f.create_group('data')
            f['data'].create_dataset(name=block, data=data)
            f['information'].create_dataset(name='description', data=description)


def cal_nomissing_days(block_file_fp,type,left,right):
    missing=cp.ones((len(block_file_fp),2400,2400),dtype=cp.uint16)
    print(missing.shape)
    print(len(block_file_fp))
    for i in range(len(block_file_fp)):
        data=read_raw_h5(block_file_fp[i])
        c1 = data < left
        c2 = data > right
        c3 = data != 65535
        data = cp.where(c1 & c3, 65535, data)
        data = cp.where(c2 & c3, 65535, data)
        data=cp.where(data!=65535,0,data)
        data=cp.where(data==65535,1,data)
        data=data.astype(cp.uint8)
        missing[i]=data
    missing=cp.sum(missing,axis=0)
    no_missing=len(type)-missing
    return no_missing


def cal_bound(mean,std):
    l=np.exp(mean-3*std)-1
    r=np.exp(mean+3*std)-1
    return l,r


if __name__ == "__main__":
    values_to_exclude = [65535]
    blocks = construct_blocks()
    values_to_exclude = [65535]
    blocks = construct_blocks()
    #read the extreme value and distribution
    extreme=pd.read_csv('./extreme/interval.csv')

    for year in range(2012, 2021):
        l, r = cal_bound(extreme.loc[year - 2012, 'mean'], extreme.loc[year - 2012, 'std'])
        print(l,r)
        left, right = cp.exp(l) - 1, cp.exp(r) - 1
        annual_holidays = dict()
        annual_works = dict()
        day_dirs = search_day_dirs(year)
        files = [search_h5_files(path) for path in day_dirs]
        daily_files = dict(zip(day_dirs, files))
        _, holidays, _, works, all = get_days(year)
        holidays_blocks = collect_block_files(year, type=holidays, annual_type=annual_holidays)
        works_blocks = collect_block_files(year, type=holidays, annual_type=annual_works)
        for key, value in holidays_blocks.items():
            print(key)
            nomissing_holidays = cal_nomissing_days(value, type=holidays,left=left,right=right)
            nomissing_holidays = nomissing_holidays.astype(cp.uint8)
            # a special situation in 2012,data for 5 days are totally missing
            if year==2012:
                nomissing_holidays=nomissing_holidays-5
            save_no_missing(data=nomissing_holidays.get(), year=year, block=key,
                            description=f"valid workdays{len(holidays)}",
                            type="holidays")
        nomissing_holidays=None
        for key, value in holidays_blocks.items():
            print(key)
            nomissing_works=cal_nomissing_days(value,type=works,left=left,right=right)
            if cp.max(nomissing_works)<=255:
                nomissing_works=nomissing_works.astype(cp.uint8)
            else:
                nomissing_works=nomissing_works.astype(cp.uint16)
            save_no_missing(data=nomissing_works.get(),year=year,block=key,
                            description=f"valid  workdays{len(works)}",
                            type="works")
        nomissing_works=None
