import numpy as np
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_hdf(files, type):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with h5py.File(path, 'r', libver='earliest') as file:
            try:
                if type == 'S2':
                    captime = pd.to_datetime(file.attrs['SENSING_TIME_GLOSDS'][:23].decode("utf-8"), yearfirst=True)
                    xdim = range(200) #int(file.attrs['NCOLS_GLOSDS'].decode("utf-8"))
                    ydim = range(200) #int(file.attrs['NROWS_GLOSDS'].decode("utf-8")))

                    R = file['/Grid/Data Fields/B04'].value[:200, :200]
                    NIR = file['/Grid/Data Fields/B05'].value[:200, :200]
                    NDVI = (NIR-R)/(NIR+R)
                    TNDVI = np.expand_dims(NDVI, axis=0)
                else:
                    captime = pd.to_datetime(file.attrs['SENSING_TIME_GLOSDS'][:27].decode("utf-8"), yearfirst=True)
                    xdim = range(200) #file.attrs['NCOLS_GLOSDS']
                    ydim = range(200) #file.attrs['NROWS_GLOSDS']

                    R = file['/Grid/Data Fields/band04'].value[:200, :200]
                    NIR = file['/Grid/Data Fields/band05'].value[:200, :200]
                    NDVI = (NIR - R) / (NIR + R)
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
    # ds_Landsat = read_hdf(r'/wad-3/Spot/HLS/ISP/L30/2016/32TMR/H5/*.h5', 'LS8')
    # ds_Sentinel = read_hdf(r'/wad-3/Spot/HLS/ISP/S30/2016/32TMR/H5/*.h5', 'S2')
    #
    # plt.figure(1)
    # ds_Sentinel[:, 100, 100].plot(style='r:')
    # ds_Landsat[:, 100, 100].plot(style='g:')
    # plt.show()


    with h5py.File("/wad-3/Spot/HLS/ISP/S30/2016/32TMR/H5/HLS.S30.T32TMR.2016003.v1.2.h5",
                   'r',
                   libver='earliest') as file:
        R = file['/Grid/Data Fields/B04'].value
        #G = file['/Grid/Data Fields/B03'].value
        #B = file['/Grid/Data Fields/B02'].value

        # R = R*((1.0/R.max().astype(np.float64)))
        # R = R.astype(np.uint)

        plt.imshow(R)
        plt.show()


    print('done')
