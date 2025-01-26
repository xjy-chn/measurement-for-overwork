import os
import time
import cupy as cp
import numpy as np
import json
import h5py
import numpy.ma as ma
import pandas as pd

def search_day_dirs(year):
    dirs = os.listdir(fr'F:\ntl\raw\{year}')
    dirs = [fr'F:\ntl\raw\{year}' + '/' + dir for dir in dirs]
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


def save(data, block, year, description):
    filename = block

    if not os.path.exists(f'./result/{year}'):
        os.makedirs(f'./result/{year}')
    with h5py.File(f'./result/{year}' + '/' + filename + '.h5', "w") as f:
        f.create_group('information')
        f.create_group('data')
        f['data'].create_dataset(name=block, data=data)
        f['information'].create_dataset(name='description', data=description)


def save_overwork(data, date, year, description, block, type, r, winsor=True):
    filename = date

    if winsor:
        if not os.path.exists(f'./result/daily_overwork/{r}x{r}_winsor/{year}/{block}/{type}'):
            os.makedirs(f'./result/daily_overwork/{r}x{r}_winsor/{year}/{block}/{type}')
        with h5py.File(f'./result/daily_overwork/{r}x{r}_winsor/{year}/{block}/{type}' + '/' + filename + '.h5',
                       "w") as f:
            f.create_group('information')
            f.create_group('data')
            f['data'].create_dataset(name=block, data=data, compression="gzip")
            f['information'].create_dataset(name='description', data=description)
    else:
        if not os.path.exists(f'./result/daily_overwork/{r}x{r}/{year}/{block}/{type}'):
            os.makedirs(f'./result/daily_overwork/{r}x{r}/{year}/{block}/{type}')
        with h5py.File(f'./result/daily_overwork/{r}x{r}/{year}/{block}/{type}' + '/' + filename + '.h5',
                       "w") as f:
            f.create_group('information')
            f.create_group('data')
            f['data'].create_dataset(name=block, data=data, compression="gzip")
            f['information'].create_dataset(name='description', data=description)


def collect_block_files(year, type, annual_type):
    for block in blocks:
        block_files = []
        for day in type:
            key = str(day['dayOfYear']).zfill(3)
            if fr'F:\ntl\raw\{year}/' + key in daily_files.keys():
                for file in daily_files[fr'F:\ntl\raw\{year}/' + key]:
                    if block in file:
                        block_files.append(fr'F:\ntl\raw\{year}/{key}/' + file)
                        break
        annual_type[block] = block_files
    return annual_type


def merge_annual_holiday_blocks(blocks_fp):

    national_annual_ntl = -cp.ones((5 * 2400, 7 * 2400), dtype=cp.uint16)
    for fp in blocks_fp:
        block = fp[-9:-3]
        h = int(block[1:3])
        v = int(block[4:6])
        with h5py.File(fp, 'r') as file:

            dataset = file['data'][block]
            data_array = cp.array(dataset)
        national_annual_ntl[2400 * (v - 3):2400 * (v - 2), 2400 * (h - 25):2400 * (h - 24)] = data_array
    return national_annual_ntl


def merge_daily_raw_blocks(blocks_fp: dict):

    daily_raw = -cp.ones((5 * 2400, 7 * 2400), dtype=cp.uint16)
    for key, fps in blocks_fp.items():
        for fp in fps:
            block = fp[17:23]
            h = int(block[1:3])
            v = int(block[4:6])
            with h5py.File(key + "/" + fp, 'r') as file:

                dataset = file['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['Gap_Filled_DNB_BRDF-Corrected_NTL']
                data_array = cp.array(dataset)
            daily_raw[2400 * (v - 3):2400 * (v - 2), 2400 * (h - 25):2400 * (h - 24)] = data_array
    return daily_raw


def cal_median(year, annual_type, left, right):

    for key, value in annual_type.items():
        daynum = len(value)
        array = cp.zeros((daynum, 2400, 2400), dtype=np.uint16)
        for i in range(daynum):
            array[i] = read_raw_h5(value[i])
        c1 = array < left
        c2 = array > right
        c3 = array != 65535
        array = cp.where(c1 & c3, 65535, array)
        array = cp.where(c2 & c3, 65535, array)
        mask = np.isin(array.get(), values_to_exclude)

        median_value = ma.median(ma.array(array.get(), mask=mask), axis=0)
        print(key, ":", median_value.shape)
        description = f"{key}'s median value in {year}"
        save_hollilday_median(data=median_value, block=key, year=year, description=description)


def save_hollilday_median(data, block, year, description):
    filename = block
    # print(filename)
    if not os.path.exists(f'./result/{year}'):
        os.makedirs(f'./result/{year}')
    with h5py.File(f'./result/{year}' + '/' + filename + '.h5', "w") as f:
        f.create_group('information')
        f.create_group('data')
        f['data'].create_dataset(name=block, data=data)
        f['information'].create_dataset(name='description', data=description)


def cal_bound(mean,std):
    l=np.exp(mean-3*std)-1
    r=np.exp(mean+3*std)-1
    return l,r


if __name__ == "__main__":
    radius = 3
    winsor = True
    values_to_exclude = [65535]
    extreme = pd.read_csv('./interval.csv')
    for year in range(2012, 2021):
        if not os.path.exists(fr'./result/daily_overwork/deep2/intensity/{year}'):
            os.makedirs(fr'./result/daily_overwork/deep2/intensity/{year}')
        if not os.path.exists(fr'./result/daily_overwork/deep2/ratio/{year}'):
            os.makedirs(fr'./result/daily_overwork/deep2/ratio/{year}')
        l, r = cal_bound(extreme.loc[year - 2012, 'mean'], extreme.loc[year - 2012, 'std'])
        print(l,r)
        left, right = cp.exp(l) - 1, cp.exp(r) - 1
        day_dirs = search_day_dirs(year)
        files = [search_h5_files(path) for path in day_dirs]
        daily_files = dict(zip(day_dirs, files))
        print(daily_files.keys())
        # get blocks and days
        blocks = sorted(construct_blocks())
        _, holidays, _, works, all = get_days(year)
        works_num = len(works)
        days = len(all)
        annual_holidays = dict()
        annual_works = dict()
        # holidays median
        annual_holidays=collect_block_files(year,type=holidays,annual_type=annual_holidays)
        cal_median(year,annual_type=annual_holidays,left=left,right=right)
        # difference
        annual_works = collect_block_files(year, type=works, annual_type=annual_works)
        print(annual_works.keys(), daily_files.keys())

        median_fp = f"./result/{year}"
        median_blocks_fp = os.listdir(median_fp)
        median_blocks_fp = sorted([median_fp + '/' + fp for fp in median_blocks_fp])
        national_annual_median_holiday_ntl = merge_annual_holiday_blocks(median_blocks_fp)

        print(len(all))
        for i in range(len(all)):
            date = str(all[i]['dayOfYear']).zfill(3)
            print(date)
            key = fr'F:\ntl\raw\{year}/{date}'
            if key in daily_files.keys():
                fp = daily_files[key]
                fp = {key: fp}
                print(fp)
                try:
                    daily_raw = merge_daily_raw_blocks(fp)
                except OSError:
                    pass
                daily_raw = daily_raw.astype(cp.int16)
                c1 = daily_raw < left
                c2 = daily_raw > right
                c3 = daily_raw != 65535
                daily_raw = cp.where(c1 & c3, 65535, daily_raw)
                daily_raw = cp.where(c2 & c3, 65535, daily_raw)
                national_annual_median_holiday_ntl = national_annual_median_holiday_ntl.astype(cp.int16)
                dif = daily_raw - national_annual_median_holiday_ntl
                dif = cp.where(daily_raw < national_annual_median_holiday_ntl, 0, dif)

                # input c2
                with h5py.File(f'./predict/{year}/{date}.h5', 'r') as file:
                    label = cp.array(file['data']['label'][:])
                    y_hat = cp.array(file['data']['predict'][:])
                    x_axis = list(file['data']['x'][:])
                    y_axis = list(file['data']['y'][:])
                deep_dummy = y_hat <= label
                deep_dummy=deep_dummy.astype(cp.uint8)
                deep_dummy = cp.where(label == 65535, 65535, deep_dummy)


                valid_dif=dif[x_axis,y_axis]
                v_national_annual_median_holiday_ntl= national_annual_median_holiday_ntl[x_axis,y_axis]
                v_daily_raw=daily_raw[x_axis,y_axis]
                valid_dif=valid_dif*deep_dummy
                valid_dif=cp.where(deep_dummy==65535,65535,valid_dif)
                valid_dif = cp.where(v_daily_raw == 65535, 65535, valid_dif)
                valid_dif = cp.where(v_national_annual_median_holiday_ntl == 65535, 65535, valid_dif)
                condition1 = valid_dif > 0
                condition2 = valid_dif != 65535
                overwork_dummy = cp.where(condition1 & condition2, 1, valid_dif)
                overwork_dummy = cp.where(overwork_dummy == 65535, 2, overwork_dummy)
                print('max',overwork_dummy.shape,cp.max(overwork_dummy),cp.count_nonzero(overwork_dummy==1)/cp.count_nonzero(overwork_dummy!=2))
                overwork_dummy = overwork_dummy.astype(cp.uint8)
                with h5py.File(fr'./result/daily_overwork/deep2/ratio/{year}/{date}.h5','w') as f:
                    f.create_group('data')
                    f['data'].create_dataset(name='dummy',data=overwork_dummy.get(),compression='gzip')
                    f['data'].create_dataset(name='x', data=np.array(x_axis), compression='gzip')
                    f['data'].create_dataset(name='y', data=np.array(y_axis), compression='gzip')
                with h5py.File(fr'./result/daily_overwork/deep2/intensity/{year}/{date}.h5','w') as f:
                    f.create_group('data')
                    f['data'].create_dataset(name='intensity',data=valid_dif.get(),compression='gzip')
                    f['data'].create_dataset(name='x', data=np.array(x_axis), compression='gzip')
                    f['data'].create_dataset(name='y', data=np.array(y_axis), compression='gzip')
