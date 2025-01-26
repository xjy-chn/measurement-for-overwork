import os
import time
import pandas as pd
import h5py
import json
import numpy as np
import numpy.ma as ma
import cupy as cp

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
    festivals, holidays, weekends, works, all = [], [], [], [], []
    if data_year != 2012:
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





def read_raw_h5(fp, block):
    if os.path.isfile(fp):
        with h5py.File(fp, 'r') as file:
            dataset = file['data'][block]
            data_array = cp.array(dataset)
    else:
        data_array = -cp.ones((2400, 2400), dtype=cp.uint16)
    return data_array


def save_annual_overwork(data, year, description, block, type):
    filename = block
    # print(filename)
    if not os.path.exists(f'./result/annual_overwork/{year}/{type}'):
        os.makedirs(f'./result/annual_overwork/{year}/{type}')
    with h5py.File(f'./result/annual_overwork/{year}/{type}' + '/' + filename + '.h5', "w") as f:
        f.create_group('information')
        f.create_group('data')
        f['data'].create_dataset(name=block, data=data,compression="gzip")
        f['information'].create_dataset(name='description', data=description)
def save_no_missing(data, year, description, block):
    filename = block
    # print(filename)
    if not os.path.exists(f'./result/overwork_nomissing/{year}'):
        os.makedirs(f'./result/overwork_nomissing/{year}')
    with h5py.File(f'./result/overwork_nomissing/{year}' + '/' + filename + '.h5', "w") as f:
        f.create_group('information')
        f.create_group('data')
        f['data'].create_dataset(name=block, data=data,compression="gzip")
        f['information'].create_dataset(name='description', data=description)
def cal_bound(mean,std):
    l=np.exp(mean-3*std)-1
    r=np.exp(mean+3*std)-1
    return l,r


if __name__ == "__main__":

    radius=5
    extreme = pd.read_csv('./interval.csv')
    for year in range(2012, 2021):
        l, r = cal_bound(extreme.loc[year - 2012, 'mean'], extreme.loc[year - 2012, 'std'])
        left, right = cp.exp(l) - 1, cp.exp(r) - 1
        types = ["intensity", "dummy"]
        _, holidays, _, works, all = get_days(year)
        values_to_exclude = [65535]

        blocks = construct_blocks()
        # merge workdays' overwork
        print(works)
        days = []
        for day in works:
            days.append(str(day['dayOfYear']).zfill(3))
        days_inten_fp = [f'./result/daily_overwork/deep2/intensity/{year}/{fp}.h5' for fp in days]
        days_dummy_fp = [f'./result/daily_overwork/deep2/ratio/{year}/{fp}.h5' for fp in days]
        with h5py.File(days_dummy_fp[0],'r') as f:
            print(f['data'].keys())
            print(f['data']['dummy'][:].shape[0])
            data = 2 * cp.ones((len(days_dummy_fp),f['data']['dummy'][:].shape[0]), dtype=cp.uint8)
            x=list(f['data']['x'][:])
            y=list(f['data']['y'][:])

        for i in range(len(days_dummy_fp)):
            if os.path.isfile(days_dummy_fp[i]):
                with h5py.File(days_dummy_fp[i],'r') as f:
                    data[i,:]=cp.array(f['data']['dummy'][:])
            missing = cp.count_nonzero(data == 2, axis=0)

        CN_works_no_missing=cp.ones((5*2400,7*2400),dtype=cp.uint16)
        for block in blocks:
            with h5py.File(f'result/overwork_nomissing/winsor/deep2/{year}/works/{block}.h5') as f:
                CN_works_no_missing[(int(block[4:6])-3)*2400:(int(block[4:6])-2)*2400,(int(block[1:3])-25)*2400:(int(block[1:3])-24)*2400] = cp.array(f['data'][block][:], dtype=cp.uint16)
        CN_works_no_missing=CN_works_no_missing[x,y]

        CN_holidays_no_missing=cp.ones((5*2400,7*2400),dtype=cp.uint16)
        for block in blocks:
            with h5py.File(f'result/overwork_nomissing/winsor/deep2/{year}/holidays/{block}.h5') as f:
                CN_holidays_no_missing[(int(block[4:6])-3)*2400:(int(block[4:6])-2)*2400,(int(block[1:3])-25)*2400:(int(block[1:3])-24)*2400] = cp.array(f['data'][block][:], dtype=cp.uint16)
                print(block[4:6],block[1:3])


        CN_holidays_no_missing=CN_holidays_no_missing[x,y]

        no_missing = len(works) - missing

        x1 = cp.where(no_missing == 0)

        overwork_days = cp.count_nonzero(data == 1, axis=0)
        ratio = overwork_days / no_missing
        ratio = np.round(ratio.get(), decimals=2)
        ratio = cp.array((100 * ratio),dtype=cp.uint8)

        ratio=cp.where(no_missing==0,101,ratio)

        ratio=cp.where( CN_works_no_missing <70, 101, ratio)

        ratio=cp.where( CN_holidays_no_missing <5, 101, ratio)


        with h5py.File(fr'./result/annual_overwork/deep2/{year}_dummy.h5','w') as f:
            f.create_group('data')
            f['data'].create_dataset(name='x',data=x,compression='gzip')
            f['data'].create_dataset(name='y', data=y, compression='gzip')
            f['data'].create_dataset(name='dummy', data=ratio.get(), compression='gzip')

