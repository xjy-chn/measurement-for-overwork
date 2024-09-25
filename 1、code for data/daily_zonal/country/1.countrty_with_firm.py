import os.path
import time

# import arcpy
import h5py
import numpy as np
from rasterstats import zonal_stats


def create_gdb(project_fp, databasename):
    if not os.path.exists(project_fp + '\\' + databasename + '.gdb'):
        gdb = arcpy.CreateFileGDB_management(project_fp, databasename)
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


if __name__ == "__main__":
    project_fp = 'F:\popLight\日度灯光分区统计'
    database_name = 'daily_zontal'
    # year=2012
    blocks = construct_blocks()
    radius = 3
    # 初始化工作路径
    if not os.path.exists(project_fp):
        os.makedirs(project_fp)
    arcpy.env.workspace = project_fp
    # 创建数据库
    create_gdb(project_fp, database_name)

    # 设置是否覆盖
    arcpy.env.overwriteOutput = True
    for year in range(2012, 2021):
        # 导入企业栅格
        with h5py.File(rf'./年度企业存量栅格/{year}firms_position.h5', 'r') as f:
            firm = np.array(f['data'][str(year)][:])
        # 拼合日度加班情况
        days = os.listdir(f'./{year}')
        days = [day for day in days if day[-4:] != '.csv']
        for day in days:
            print('day', day)
            dummy = 2 * np.ones((5 * 2400, 7 * 2400), dtype=np.uint8)
            for v in range(5):
                for h in range(7):
                    inten_fp = f'./result/daily_overwork/{radius}x{radius}_winsor/{year}/h{h + 25}v{str(v + 3).zfill(2)}/dummy/{day}.h5'
                    # print('a',inten_fp)
                    if os.path.isfile(inten_fp):
                        with h5py.File(inten_fp, "r") as f:
                            data = f["data"][f'h{h + 25}v{str(v + 3).zfill(2)}'][:]
                        dummy[v * 2400:(v + 1) * 2400, h * 2400:(h + 1) * 2400] = data
            # raster=arcpy.NumPyArrayToRaster(dummy,arcpy.Point(70, 10),
            #                                   x_cell_size=0.004166666666666667, y_cell_size=0.004166666666666667,
            #                                   value_to_nodata=2)
            dummy = np.where(firm == 0, 2, dummy)
            arcpy.sa.ZonalStatisticsAsTable(r"F:\日度夜间灯光\原始数据\2019年中国各级行政区划\v84\国.shp",
                                            "FID", arcpy.NumPyArrayToRaster(dummy, arcpy.Point(70, 10),
                                                                            x_cell_size=0.004166666666666667,
                                                                            y_cell_size=0.004166666666666667,
                                                                            value_to_nodata=2),
                                            rf"F:\popLight\日度灯光分区统计\daily_zontal.gdb\sfn{year}{day}", "DATA",
                                            "ALL", "CURRENT_SLICE", 90, "AUTO_DETECT", "ARITHMETIC", 360)
            print(f"第{year}年第{day}天加班分区统计已输出完毕")
        # time.sleep(1000)
    arcpy.env.overwriteOutput = False
