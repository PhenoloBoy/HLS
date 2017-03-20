import numpy as np
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import argparse
import numpy as np
import pandas as pd
import rasterio as rs
import phen
from netCDF4 import Dataset, date2num
from datetime import datetime
import output



def read_hdf(files, type, coords):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with h5py.File(path, 'r', libver='earliest') as file:
            try:
                if type == 'S2':
                    rb_name = '/Grid/Data Fields/B04'
                    nir_name = '/Grid/Data Fields/B8A'
                    cut_pos = 23
                else:
                    rb_name = '/Grid/Data Fields/band04'
                    nir_name = '/Grid/Data Fields/band05'
                    cut_pos = 27

                xdim = range(coords[0], coords[0] + 1)
                ydim = range(coords[1], coords[1] + 1)

                captime = pd.to_datetime(file.attrs['SENSING_TIME_GLOSDS'][:cut_pos].decode("utf-8"), yearfirst=True)

                R = file[rb_name].value[coords[0]: coords[0]+1, coords[1]: coords[1]+1].astype('f8')
                R[R == -1000] = np.NaN
                NIR = file[nir_name].value[coords[0]: coords[0]+1, coords[1]: coords[1]+1].astype('f8')
                NIR[NIR == -1000] = np.NaN
                NDVI = (NIR-R)/(NIR+R)
                TNDVI = np.expand_dims(NDVI, axis=0)

                try:
                    singleton = pd.Panel(TNDVI, items=[captime], major_axis=ydim, minor_axis=xdim)
                except():
                    pass

                return singleton
            except():
                pass

    paths = sorted(glob.glob(files))

    concat = pd.concat([process_one_path(p) for p in paths])

    return concat


if __name__ == '__main__':

    x = 703
    y = 2692

    coords = [x, y]

    ds_Landsat = read_hdf(r'G:\ISP\L30\H5\*.h5', 'LS8', coords)
    ds_Sentinel = read_hdf(r'G:\ISP\S30\H5\*.h5', 'S2', coords)
    ds_tot = ds_Sentinel.join(ds_Landsat)[:, y, x]

    ts_table = phen.analyzes(ds_tot, coords)


    # plt.figure(1)
    # plt.ioff()
    # ds_Sentinel[:, y, x].interpolate().plot(style='r:')
    # ds_Landsat[:, y, x].interpolate().plot(style='g:')
    # ds_Sentinel[:, y, x].plot(style='ro')
    # ds_Landsat[:, y, x].plot(style='go')
    # plt.show()

    # with h5py.File(r"G:\ISP\S30\2016\32TMR\H5\HLS.S30.T32TMR.2016003.v1.2.h5",
    #                'r',
    #                libver='earliest') as file:
    #     R = file['/Grid/Data Fields/B04'].value
    #     G = file['/Grid/Data Fields/B03'].value
    #     B = file['/Grid/Data Fields/B02'].value
    #
    #     plt.imshow(R)
    #     plt.imshow(G)
    #     plt.imshow(B)
    #     plt.show()

    print('done')
