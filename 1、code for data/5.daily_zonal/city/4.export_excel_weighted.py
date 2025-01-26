import os.path
import time

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
if __name__=="__main__":
    project_fp = 'F:\overwork'
    database_name = 'daily_zontal'
    blocks = construct_blocks()
    radius=5


    arcpy.env.workspace = project_fp

    arcpy.env.overwriteOutput=False
    for year in range(2012,2021):
        if not os.path.exists(rf"./zonal/{radius}x{radius}\city_weighted\{year}"):
            os.makedirs(rf"./zonal/{radius}x{radius}\city_weighted\{year}")
        days = os.listdir(fr'F:\ntl\raw\{year}')
        days=[day for day in days if day[-4:]!='.csv']
        print(days)
        for day in days:
            arcpy.conversion.TableToExcel(rf"F:/overwork/daily_zontal.gdb/deep2_swfc{year}{day}",
                                          rf"./zonal/{radius}x{radius}\city_weighted\{year}\{day}.xlsx", "NAME", "CODE")
            print(f"zonal at the {day} in in {year} finished")
    arcpy.env.overwriteOutput = False



