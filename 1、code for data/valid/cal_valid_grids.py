import os
import time

import numpy as np
import json
import h5py
import numpy.ma as ma
import cupy as cp

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
    if data_year != 2012:
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
    elif data_year == 2012:
        with open(f'./万年历/{data_year}.json', 'r', encoding='utf-8') as file:
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


def save_no_missing(data, year, block, description,type,bound=True):
    # print(filename)
    if bound:
        if not os.path.exists(f'./result/overwork_nomissing/winsor/{year}/{type}'):
            os.makedirs(f'./result/overwork_nomissing/winsor/{year}/{type}')
        with h5py.File(f'./result/overwork_nomissing/winsor/{year}/{type}' + '/' + block + '.h5', "w") as f:
            f.create_group('imformation')
            f.create_group('data')
            f['data'].create_dataset(name=block, data=data)
            f['imformation'].create_dataset(name='基本信息', data=description)
    else:
        if not os.path.exists(f'./result/overwork_nomissing/{year}/{type}'):
            os.makedirs(f'./result/overwork_nomissing/{year}/{type}')
        with h5py.File(f'./result/overwork_nomissing/{year}/{type}' + '/' + block + '.h5', "w") as f:
            f.create_group('imformation')
            f.create_group('data')
            f['data'].create_dataset(name=block, data=data)
            f['imformation'].create_dataset(name='基本信息', data=description)


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
    # x, y, z = cp.where(missing==0)
    missing=cp.sum(missing,axis=0)
    no_missing=len(type)-missing
    return no_missing
    # print(no_missing.shape)
    # print(no_missing)
    # print(cp.max(no_missing))
    # print(cp.min(no_missing))
    # print(cp.mean(missing)/len(type))
        # print(key + '/' + block)
        # data = read_raw_h5(key + '/' + block)

def cal_bound(year):
    with h5py.File(f'./计算正态区间/{year}.h5', 'r') as f:
        mean = cp.array(f['data']['mean'])
        variance = cp.array(f['data']['std'])
        valid = cp.array(f['data']['vaid_grid'])
        sum = cp.nansum(mean * valid)
        total_variance = cp.nansum(variance * valid)
        total_valid = cp.nansum(valid)
        mean = sum / total_valid
        std = (total_variance / total_valid) ** 0.5
        return float(mean - 3 * std), float(mean + 3 * std)


if __name__ == "__main__":
    values_to_exclude = [65535]
    blocks = construct_blocks()
    # for year in range(2012,2025):
    #     if year not in [2012,2022,2024]:
    for year in range(2012, 2021):
        l, r = cal_bound(year)
        left, right = cp.exp(l) - 1, cp.exp(r) - 1
        annual_holidays = dict()
        annual_works = dict()
        day_dirs = search_day_dirs(year)
        files = [search_h5_files(path) for path in day_dirs]
        daily_files = dict(zip(day_dirs, files))
        _, holidays, _, works, all = get_days(year)

        holidays_blocks = collect_block_files(year, type=holidays, annual_type=annual_holidays)
        works_blocks = collect_block_files(year, type=holidays, annual_type=annual_works)
        # print(holidays_blocks.keys())
        # print(works_blocks.keys())
        for key, value in holidays_blocks.items():
            print(key)
            nomissing_holidays = cal_nomissing_days(value, type=holidays,left=left,right=right)
            # nomissing_holidays=nomissing_holidays<5
            nomissing_holidays = nomissing_holidays.astype(cp.uint8)
            if year==2012:
                nomissing_holidays=nomissing_holidays-5
            save_no_missing(data=nomissing_holidays.get(), year=year, block=key,
                            description=f"这是节假日有原始数据的天数，节假日总天数为{len(holidays)}",
                            type="holidays")
        nomissing_holidays=None
        for key, value in holidays_blocks.items():
            print(key)
            nomissing_works=cal_nomissing_days(value,type=works,left=left,right=right)
            # nomissing_works=nomissing_works>100
            if cp.max(nomissing_works)<=255:
                nomissing_works=nomissing_works.astype(cp.uint8)
            else:
                nomissing_works=nomissing_works.astype(cp.uint16)
            save_no_missing(data=nomissing_works.get(),year=year,block=key,
                            description=f"这是工作日有原始数据的天数，节假日总天数为{len(works)}",
                            type="works")
        nomissing_works=None


        # cal_surrounding_median_dummy()
