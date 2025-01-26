import os.path
import time
# import cupy as cp
# import arcpy
import h5py
import numpy as np
import cupy as cp

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
    radius=5
    blocks=construct_blocks()

    # initial work path
    if not os.path.exists(project_fp):
        os.makedirs(project_fp)

    for year in range(2012,2021):
    # input the number of firms at grids level
        with h5py.File(rf'./{year}firms_position.h5', 'r') as f:
            firm = np.array(f['data'][str(year)][:])
        if not os.path.exists(f'./result/deep2/firm_weight/{radius}x{radius}/{year}'):
            os.makedirs(f'./result/deep2/firm_weight/{radius}x{radius}/{year}')
    # get grids index from overwork
        days = os.listdir(fr'F:\ntl\raw\{year}')
        days=[day for day in days if day[-4:]!='.csv']
        print(days)
        with h5py.File(f'./result/daily_overwork/deep2/ratio/{year}/{days[0]}.h5', 'r') as f:
            x_axis = f['data']['x'][:]
            y_axis = f['data']['y'][:]
        firm = firm[x_axis, y_axis]
    # index include some listed firm workplace from overwork, when the number of firm=0
        for day in days:
            with h5py.File(f'./result/daily_overwork/deep2/ratio/{year}/{day}.h5','r') as f:
                x_axis=f['data']['x'][:]
                y_axis = f['data']['y'][:]
                dummy= f['data']['dummy'][:]
            dummy2=dummy/firm
            dummy2=100*np.round(dummy2,decimals=2).astype(np.uint8)
            dummy2=np.where(dummy==2,200,dummy2)
            print(dummy2,dummy,firm)
            time.sleep(100)
            with h5py.File(f'./result/deep2/firm_weight/{radius}x{radius}/{year}/{day}.h5 ','w') as f:
                dataset=f.create_group('data')
                f['data'].create_dataset(name=day, data=dummy2,compression="gzip")
                f['data'].create_dataset(name='x', data=x_axis, compression="gzip")
                f['data'].create_dataset(name='y', data=y_axis, compression="gzip")
                f['data'].create_dataset(name='nan',data=200)



