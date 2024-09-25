import os.path
import time

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
    project_fp = 'F:\popLight\日度灯光分区统计'
    database_name = 'daily_zontal'
    radius=3
    # year=2012
    blocks=construct_blocks()

    # #初始化工作路径
    # if not os.path.exists(project_fp):
    #     os.makedirs(project_fp)
    arcpy.env.workspace = project_fp
    #创建数据库
    # create_gdb(project_fp,database_name)

    #设置是否覆盖
    arcpy.env.overwriteOutput=False
    for year in range(2012,2021):
        if not os.path.exists(rf"F:\日度夜间灯光\原始数据\result\日度分区统计\deep\{radius}x{radius}\分城市有企业\{year}"):
            os.makedirs(rf"F:\日度夜间灯光\原始数据\result\日度分区统计\deep\{radius}x{radius}\分城市有企业\{year}")
    #拼合日度加班情况
        days = os.listdir(fr'F:\日度夜间灯光\原始数据\{year}')
        days=[day for day in days if day[-4:]!='.csv']
        print(days)
        for day in days:
            # arcpy.sa.ZonalStatisticsAsTable(r"F:\日度夜间灯光\原始数据\2019年中国各级行政区划\v84\市_WGCS1984.shp",
            #                                 "市代码", arcpy.NumPyArrayToRaster(dummy, arcpy.Point(70, 10),
            #                                                                    x_cell_size=0.004166666666666667,
            #                                                                    y_cell_size=0.004166666666666667,
            #                                                                    value_to_nodata=2),
            #                                 rf"F:\popLight\日度灯光分区统计\daily_zontal.gdb\z{year}{day}", "DATA",
            #                                 "ALL", "CURRENT_SLICE", 90, "AUTO_DETECT", "ARITHMETIC", 360)
            arcpy.conversion.TableToExcel(rf"F:\popLight\日度灯光分区统计\daily_zontal.gdb\deep_sfc{year}{day}",
                                          rf"F:\日度夜间灯光\原始数据\result\日度分区统计\deep\{radius}x{radius}\分城市有企业\{year}\{day}.xlsx", "NAME", "CODE")
            print(f"第{year}年第{day}天加班分区统计excel已输出完毕")
        # time.sleep(1000)
    arcpy.env.overwriteOutput = False



