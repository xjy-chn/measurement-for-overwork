
import os
import time

import pandas as pd
if __name__=="__main__":
    radius=3
    for year in range(2012,2021):
        data = pd.DataFrame()
        annual_files=sorted(os.listdir(rf"./zonal/{radius}x{radius}/city/{year}"))
        abs_Afiles_fp=sorted([rf"./zonal/{radius}x{radius}/city/{year}"+'/'+file for file in annual_files])
        print(annual_files)
        print(abs_Afiles_fp)
        for i in range(len(abs_Afiles_fp)):
            daily_data=pd.read_excel(abs_Afiles_fp[i])
            daily_data['year']=year
            daily_data['dayOfYear']=annual_files[i][0:3]
            data=pd.concat([data,daily_data])
        data.to_excel(rf'./zonal/{radius}x{radius}/city/{year}statistic.xlsx',index=False)

