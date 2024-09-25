import os.path
import time
# import cupy as cp
import arcpy
import h5py
import numpy as np

def create_gdb(project_fp,databasename):
    if not os.path.exists(project_fp+'\\'+databasename+'.gdb'):
        gdb=arcpy.CreateFileGDB_management(project_fp,databasename)
        print(type(gdb))
        print("数据库创建成功")
    else:
        print("数据库在之前已创建")
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
def merge_blocks():
    pass
if __name__=="__main__":
    time.sleep(180)
    project_fp = 'F:\popLight\日度灯光分区统计'
    database_name = 'daily_zontal'
    # year=2012
    blocks=construct_blocks()

    #初始化工作路径
    if not os.path.exists(project_fp):
        os.makedirs(project_fp)
    arcpy.env.workspace = project_fp
    #创建数据库
    create_gdb(project_fp,database_name)

    #设置是否覆盖
    arcpy.env.overwriteOutput=True
    for year in range(2014,2021):
    # #导入企业栅格
    #     with h5py.File(rf'./年度企业存量栅格/{year}firms_position.h5', 'r') as f:
    #         firm = cp.array(f['data'][str(year)][:])
    #     if not os.path.exists(f'./result/日度分区统计/企业加权暂存/{year}'):
    #         os.makedirs(f'./result/日度分区统计/企业加权暂存/{year}')
    #拼合日度加班情况
        days = os.listdir(f'./{year}')
        days=[day for day in days if day[-4:]!='.csv']
        c,i=0,0
        for day in days:
            # print('day',day)
            # dummy = 2 * cp.ones((5 * 2400, 7 * 2400), dtype=cp.uint8)
            # for v in range(5):
            #     for h in range(7):
            #         inten_fp = f'./result/daily_overwork/{year}/h{h + 25}v{str(v + 3).zfill(2)}/dummy/{day}.h5'
            #         print('a',inten_fp)
            #         if os.path.isfile(inten_fp):
            #             with h5py.File(inten_fp, "r") as f:
            #                 data = cp.array(f["data"][f'h{h + 25}v{str(v + 3).zfill(2)}'][:])
            #             dummy[v * 2400:(v + 1) * 2400, h * 2400:(h + 1) * 2400] = data
            # # raster=arcpy.NumPyArrayToRaster(dummy,arcpy.Point(70, 10),
            # #                                   x_cell_size=0.004166666666666667, y_cell_size=0.004166666666666667,
            # #                                   value_to_nodata=2)
            # firm2=1/firm
            # firm2=cp.where(firm==0,0,firm2)
            # dummy2=dummy*firm2
            # dummy2=cp.array(np.round((100*dummy2)).get())
            # dummy=cp.where(firm==0,2,dummy)
            # dummy2 = cp.where(dummy == 2, 200, dummy2)
            # dummy2 = cp.where(firm == 0, 200, dummy2).astype(cp.uint8)
            #
            # with h5py.File(f'./result/日度分区统计/企业加权暂存/{year}/{day}.h5 ','w') as f:
            #     dataset=f.create_group('data')
            #     f['data'].create_dataset(name=day, data=dummy2.get())
            #     f['data'].create_dataset(name='nan',data=200)
            st=time.time()
            with h5py.File(f'./result/日度分区统计/deep/企业加权暂存/3x3/{year}/{day}.h5 ', 'r') as f:
                dummy2=200*np.ones((12000,16800),dtype=np.uint8)
                x=f['data']['x'][:]
                y = f['data']['y'][:]
                dummy2[x,y]= f['data'][day][:]
                # print(dummy2.shape)
                # print(np.count_nonzero(dummy2==200))
                # time.sleep(100)

            arcpy.sa.ZonalStatisticsAsTable(r"F:\日度夜间灯光\原始数据\2019年中国各级行政区划\v84\国.shp",
                                            "FID", arcpy.NumPyArrayToRaster(dummy2, arcpy.Point(70, 10),
                                                                            x_cell_size=0.004166666666666667,
                                                                            y_cell_size=0.004166666666666667,
                                                                            value_to_nodata=200),
                                            rf"F:\popLight\日度灯光分区统计\daily_zontal.gdb\deep_swfn{year}{day}", "DATA",
                                            "ALL", "CURRENT_SLICE", 90, "AUTO_DETECT", "ARITHMETIC", 360)
            # arcpy.sa.ZonalStatisticsAsTable(r"F:\日度夜间灯光\原始数据\2019年中国各级行政区划\v84\国.shp",
            #                                 "FID", arcpy.NumPyArrayToRaster(dummy2, arcpy.Point(70, 10),
            #                                                                 x_cell_size=0.004166666666666667,
            #                                                                 y_cell_size=0.004166666666666667,
            #                                                                 value_to_nodata=200),
            #                                 rf"F:\popLight\日度灯光分区统计\daily_zontal.gdb\deep_swfn{year}{day}",
            #                                 "DATA",
            #                                 "ALL", "CURRENT_SLICE", 90, "AUTO_DETECT", "ARITHMETIC", 360)
            print(f"第{year}年第{day}天加班分区统计已输出完毕")
            c+=time.time()-st
            i+=1
            avg=c/i
            print(avg,c)
        # time.sleep(1000)
    arcpy.env.overwriteOutput = False



