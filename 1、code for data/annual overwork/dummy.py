import os
import time

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


def scan_all_files():
    for key, value in fps.items():
        block_inten_fp_f = os.listdir(f'./result/daily_overwork/{year}/{key}/intensity')
        block_inten_fp = [f'./result/daily_overwork/{year}/{key}/intensity' + '/' + fp for fp in block_inten_fp_f]
        block_dummy_fp = [f'./result/daily_overwork/{year}/{key}/dummy' + '/' + fp for fp in block_inten_fp_f]
        value['intensity'] = block_inten_fp
        value['dummy'] = block_dummy_fp
    print(fps['h25v03']['intensity'])


def read_raw_h5(fp, block):
    if os.path.isfile(fp):
        with h5py.File(fp, 'r') as file:
            # 读取数据集到 NumPy 数组
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
        f.create_group('imformation')
        f.create_group('data')
        f['data'].create_dataset(name=block, data=data,compression="gzip")
        f['imformation'].create_dataset(name='基本信息', data=description)
def save_no_missing(data, year, description, block):
    filename = block
    # print(filename)
    if not os.path.exists(f'./result/overwork_nomissing/{year}'):
        os.makedirs(f'./result/overwork_nomissing/{year}')
    with h5py.File(f'./result/overwork_nomissing/{year}' + '/' + filename + '.h5', "w") as f:
        f.create_group('imformation')
        f.create_group('data')
        f['data'].create_dataset(name=block, data=data,compression="gzip")
        f['imformation'].create_dataset(name='基本信息', data=description)
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

    # -----------------3、拼接日度数据计算年度指标dummy.py
    radius=3
    for year in range(2012, 2021):
        l, r = cal_bound(year)
        left, right = cp.exp(l) - 1, cp.exp(r) - 1
        types = ["intensity", "dummy"]
        _, holidays, _, works, all = get_days(year)
        values_to_exclude = [65535]

        # -----------------------
        blocks = construct_blocks()
        blocks_fp = sorted(os.listdir(f'./result/daily_overwork/{radius}x{radius}_winsor/{year}'))
        blocks_fp = sorted([f'./result/daily_overwork/{radius}x{radius}_winsor/{year}' + '/' + block for block in blocks_fp])

        fps = dict()
        for i in range(len(blocks)):
            fps[blocks[i]] = {
                "intensity": [],
                "dummy": []
            }
        # 确定合并范围：暂时只考虑工作日
        print(works)
        days = []
        for day in works:
            days.append(str(day['dayOfYear']).zfill(3))

        for block, value in fps.items():
            # print(block)
            days_inten_fp = [f'./result/daily_overwork/{radius}x{radius}_winsor/{year}/{block}/intensity' + '/' + fp + '.h5' for fp in days]
            days_dummy_fp = [f'./result/daily_overwork/{radius}x{radius}_winsor/{year}/{block}/dummy' + '/' + fp + '.h5' for fp in days]
            fps[block]['intensity'] = days_inten_fp
            fps[block]['dummy'] = days_dummy_fp
        for block, value in fps.items():
            # print(block,value['dummy'])
            # time.sleep(100)
            st = time.time()
            num = len(value['dummy'])
            data = 2*cp.ones((num, 2400, 2400), dtype=cp.uint8)
            for i in range(num):
                data[i] = read_raw_h5(value['dummy'][i], block)
            missing = cp.count_nonzero(data == 2, axis=0)

            no_missing = len(works) - missing
            x1,y1=cp.where(no_missing==0)
            print(x1,y1)
            print(np.min(no_missing),np.max(no_missing))
            overwork_days = cp.count_nonzero(data==1, axis=0)
            ratio = overwork_days / no_missing
            #导入缺失天数情况
            with h5py.File(f'result/overwork_nomissing/winsor/{year}/works/{block}.h5') as f:
                works_no_missing=cp.array(f['data'][block][:],dtype=cp.uint16)
            with h5py.File(f'result/overwork_nomissing/winsor/{year}/holidays/{block}.h5') as f:
                holidays_no_missing = cp.array(f['data'][block][:],dtype=cp.uint16)

            # 增加导出非工作日no_missing数据的代码
            ratio = np.round(ratio.get(), decimals=2)

            ratio = cp.array((100 * ratio),dtype=cp.uint8)
            print('基础缺失数据', cp.count_nonzero(ratio == 101))
            ratio = cp.where(no_missing == 0, 101, ratio)
            print('全年缺失数据', cp.count_nonzero(ratio == 101))
            ratio=cp.where(works_no_missing <70, 101, ratio)
            print('工作日基础缺失数据', cp.count_nonzero(ratio == 101))
            ratio=cp.where(holidays_no_missing <5, 101, ratio)
            print('节假日基础缺失数据', cp.count_nonzero(ratio == 101),cp.count_nonzero(holidays_no_missing <5),holidays_no_missing.shape)
            time.sleep(1)
            # print(ra)
            print(cp.min(works_no_missing),cp.min(holidays_no_missing))
            print(cp.max(ratio),ratio.dtype)
            # if block=="h31v06":
            print(no_missing)
            print(works_no_missing)
            print(holidays_no_missing)
            print(cp.max((works_no_missing + holidays_no_missing >= no_missing)))
            print(ratio)
            # save_annual_overwork(data=ratio.get(), year=year,
            #                      description=f"{year}年{block}块的年度加班天数占比,数值四舍五入后放大了100倍,缺失值为101",
            #                      block=block, type="ratio")


            intensity = None
            ratio = None
            data=None
            mask=None
            condition2=None
            condition1=None
            print(f"{year}年{block}保存完成")
            print(f"用时{time.time() - st}秒")

# {block:{intensity:[],dummy:[]}
