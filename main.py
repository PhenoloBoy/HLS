import numpy as np
import glob
import h5py
import pandas as pd


def read_hdf(files):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with h5py.File(path, 'r', libver='earliest') as file:

            captime = pd.to_datetime(file.attrs['SENSING_TIME_GLOSDS'].decode("utf-8"), yearfirst=True)
            xdim = range(int(file.attrs['NCOLS_GLOSDS'].decode("utf-8")))
            ydim = range(int(file.attrs['NROWS_GLOSDS'].decode("utf-8")))

            R = file['/Grid/Data Fields/B04'].value
            NIR = file['/Grid/Data Fields/B05'].value
            NDVI = (NIR-R)/(NIR+R)
            TNDVI = np.expand_dims(NDVI, axis=0)

            singleton = pd.Panel(TNDVI, items=[captime], major_axis=ydim, minor_axis=xdim)
            return singleton

    paths = sorted(glob.glob(files))

    concat = pd.concat([process_one_path(p) for p in paths])

    return concat


if __name__ == '__main__':
    ds = read_hdf(r'G:\HLS\S30\2016\31UDP\h5\*.h5')
    ds[:, 0, 0].plot()