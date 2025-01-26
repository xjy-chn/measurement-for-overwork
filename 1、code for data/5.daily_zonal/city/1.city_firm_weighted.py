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
        print("database created")
    else:
        print("database has been created")
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

def merge_blocks():
    pass
if __name__=="__main__":
    project_fp = 'F:\overwork'
    database_name = 'daily_zontal'
    blocks = construct_blocks()
    radius=5
    # initial
    if not os.path.exists(project_fp):
        os.makedirs(project_fp)
    arcpy.env.workspace = project_fp
    # create database
    create_gdb(project_fp, database_name)

    arcpy.env.overwriteOutput = True
    for year in range(2012, 2021):
        #file for raw data from NASA
        days = os.listdir(fr'F:\ntl\raw\{year}')
        days = [day for day in days if day[-4:] != '.csv']
        for day in days:
            st = time.time()
            with h5py.File(f'./result/deep2/firm_weight/{radius}x{radius}/{year}/{day}.h5', 'r') as f:
                dummy2 = 200 * np.ones((12000, 16800), dtype=np.uint8)
                x = f['data']['x'][:]
                y = f['data']['y'][:]
                dummy2[x, y] = f['data'][day][:]

            arcpy.sa.ZonalStatisticsAsTable(r"./city84.shp",
                                            "citycode", arcpy.NumPyArrayToRaster(dummy2, arcpy.Point(70, 10),
                                                                               x_cell_size=0.004166666666666667,
                                                                               y_cell_size=0.004166666666666667,
                                                                               value_to_nodata=200),
                                            rf"F:\overwork\daily_zontal.gdb\deep2_swfc{year}{day}", "DATA",
                                            "ALL", "CURRENT_SLICE", 90, "AUTO_DETECT", "ARITHMETIC", 360)
            print(f"the daily zonal for data at {day} in {year} is finished")
    arcpy.env.overwriteOutput = False



