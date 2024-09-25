
import os
import time

import pandas as pd
if __name__=="__main__":
    radius=3
    for year in range(2012,2021):
        data = pd.DataFrame()
        annual_files=sorted(os.listdir(f'./result/日度分区统计/deep/{radius}x{radius}/全国企业数量加权/{year}'))
        abs_Afiles_fp=sorted([f'./result/日度分区统计/deep/{radius}x{radius}/全国企业数量加权/{year}'+'/'+file for file in annual_files])
        print(annual_files)
        print(abs_Afiles_fp)
        for i in range(len(abs_Afiles_fp)):
            daily_data=pd.read_excel(abs_Afiles_fp[i])
            daily_data['year']=year
            daily_data['dayOfYear']=annual_files[i][0:3]
            data=pd.concat([data,daily_data])
            print(f"{year}年第{annual_files[i][0:3]}天数据已合并")
        data.to_excel(f'./result/日度分区统计/deep/{radius}x{radius}/全国企业数量加权/{year}年度汇总.xlsx',index=False)
            # da
