#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import outlier
import detect_peacks as dp
from seasonal import fit_seasons


def analyzes(h0, coord):
    # Numpy error report
    np.seterr(all='ignore')  # {‘ignore’, ‘warn’, ‘raise’, ‘call’, ‘print’, ‘log’}

    # Error table
    err_table = pd.Series(0, index=['read', 'fix', 'rescaling', 'outlier', 'cleaning',
                                    's&t_ext', 's2d', 'savgol', 'valley'])

    medspan = 51
    smp = 4
    tr = 75
    mavmet = 1.5

    # Output container
    ts_table = pd.DataFrame([])

    # Convert to pandas series
    ts_dek = h0.interpolate()

    cleaned = outlier.old_clean(ts_dek, [0, 1], 1)

    cled = cleaned.rolling(window=7, center=True).mean()

    # Interpolate data to daily sample
    ts_d = cled.resample('D').asfreq().interpolate(method='linear').fillna(0)

    try:
        # Savinsky Golet filter
        ps = savgol_filter(ts_d, 30, 3, mode='nearest')
    except (RuntimeError, Exception, ValueError):
        print('Error! Savinsky Golet filter problem, in position:{0}'.format(coord))
        err_table['savgol'] = 1
        return

    # Valley detection
    try:
        #
        ind = dp.detect_peaks(ps, mph=-ps.mean()+ps.std(),
                              mpd=int(360 * tr / 100),
                              valley=True,
                              kpsh=False)
        oversmp = False
        if not ind.any():
            ind = dp.detect_peaks(ps, mph=-20,
                                  mpd=60,
                                  valley=True)
            oversmp = True

    except ValueError as e:
        print('Error in valley detection, in position:{0}, error {1}'.format(coord, e))
        err_table['vally'] = 1
        return

    # Valley point time series conversion
    pks = pd.Series()
    for i in ind:
        pks[ps.index[i]] = ps.iloc[i]

    # plt.subplot(211)
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # ts_dek_unfix.plot(style='g--')
    # ts_dek.plot(style='r--')
    #
    # plt.subplot(212)
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    # ps.plot(style='r:')
    # pks.plot(style='ro')
    # plt.show()
    # plt.clf()

    # Point to point cycles
    for i in range(len(pks) - 1):

        # Min min curve
        mmi = ps[pks.index[i]: pks.index[i + 1]]

        # Interpolate line between min min
        pf = mmi.copy()
        pf.iloc[1:-1] = np.nan
        pf = pf.interpolate()

        # Vox
        vox = mmi - pf
        integral_vox = vox.sum()

        if oversmp and integral_vox < 500:
            continue

        # Barycenter
        index = vox.index
        unx_time = index.values.astype(np.int64) // 10 ** 9
        unx_sbc = (unx_time * vox).sum() / vox.sum()

        if unx_sbc < 0:  # TODO clean the situation
            print('Warning! unx_sbc < 0, in position:{0}'.format(coord))
            continue
        else:
            try:
                sbc = pd.to_datetime(unx_sbc * 10 ** 9)
            except (RuntimeError, Exception, ValueError):
                print('Warning! Datetime conversion went wrong, in position:{0}'.format(coord))
                continue

        # avoid unusual results
        if sbc.year not in range(pks.index[i].year - 1, pks.index[i + 1].year + 1):
            print('Warning! sbc not in a valid range, in position:{0}'.format(coord))
            continue
        # Season deviation standard
        sds = np.sqrt((np.square(unx_time) * vox).sum() / vox.sum() - np.square(unx_sbc))

        # try:
        #
        # except RuntimeError:
        #     print('Season deviation standard error')
        #     return

        if np.isnan(sds):
            print('Warning! Season deviation standard is Nan, in position:{0}'.format(coord))
            continue

        sds_d = pd.Timedelta(sds, unit='s')

        # Update values
        row = pd.DataFrame([(pks.index[i], pks.index[i + 1], sbc, sds_d)],
                           columns=['sd', 'ed', 'sbc', 'ssd'],
                           index=[sbc])

        ts_table = ts_table.append(row)

    # Core
    try:
        for i in range(len(ts_table)):

            index = ts_table.iloc[i]['sbc']

            # specific mas
            mas = (ts_table.iloc[i]['ed'] - ts_table.iloc[i]['sd']) - (2 * mavmet * ts_table.iloc[i]['ssd'])

            original = ps[ts_table.iloc[i]['sd']:ts_table.iloc[i]['ed']]

            timedelta = (ts_table.iloc[i]['ed'] - ts_table.iloc[i]['sd']) * 2/3

            buffered = ps[ts_table.iloc[i]['sd']-timedelta:ts_table.iloc[i]['ed']+timedelta]

            try:
                smth_crv = buffered.rolling(mas.days, win_type='boxcar',
                                        center=True).mean()  # TODO test other sample method
            except (RuntimeError, Exception, ValueError):
                print('Warning! Smoothed curv calculation went wrong, in position:{0}'.format(coord))
                return

            smoothed = smth_crv[ts_table.iloc[i]['sd'] - timedelta:ts_table.iloc[i]['ed'] + timedelta]

            back = smoothed[:ts_table.iloc[i]['sbc']]\
                .shift(1, freq=pd.Timedelta(days=int(mas.days / 2)))\
                [ts_table.iloc[i]['sd']:].dropna()

            forward = smoothed[ts_table.iloc[i]['sbc']:]\
                .shift(1, freq=pd.Timedelta(days=-int(mas.days / 2))) \
                [:ts_table.iloc[i]['ed']].dropna()

            # original = original.reindex(pd.date_range(forward.index[0].date(), back.index[len(forward) - 1].date()))

            sbd = None
            sed = None

            sbd_ts = pd.Series()
            sed_ts = pd.Series()

            # research of the starting season
            for s in range(len(back) - 1):
                date0 = back.index[s]
                date1 = back.index[s + 1]

                if back.loc[date0] >= original.loc[date0] and back.loc[date1] < original.loc[date1]:
                    sbd = back.index[s]
                    sbd_ts.loc[back.index[s]] = back.iloc[s]
                    break  # TODO insert more than one intersection

            # research of the end seasons
            for e in range(len(forward) - 1):
                date0 = forward.index[e]
                date1 = forward.index[e + 1]

                if forward.loc[date0] <= original.loc[date0] and forward.loc[date1] > original.loc[date1]:
                    sed = forward.index[e]
                    if sed_ts.count() == 0:
                        sed_ts.loc[forward.index[e]] = forward.iloc[e]
                    else:
                        sed_ts = pd.Series()
                        sed_ts.loc[forward.index[e]] = forward.iloc[e]
                        # break
                        # TODO insert more than one intersection

            max_date = original.idxmax()

            if sed is None or sbd is None:
                continue
            else:
                # Season slope
                sslp = (sed_ts.values[0] - sbd_ts.values[0]) / (sed.date() - sbd.date()).days
                if not(-0.15 < sslp < 0.15):
                    continue

            try:
                # Start date of the season
                ts_table.set_value(index, 'sbd', np.datetime64(sbd.date()))

                # End date of the season
                ts_table.set_value(index, 'sed', np.datetime64(sed.date()))

                # Season slope
                ts_table.set_value(index, 'sslp', sslp)

                # Season Lenght
                sl = (sed - sbd).days
                ts_table.set_value(index, 'sl', sl)

                # Season permanet
                sp = sbd_ts.append(sed_ts).resample('D').asfreq().interpolate(method='linear')
                spi = sp.sum()
                ts_table.set_value(index, 'spi', spi)

                # Season Integral
                si = original.loc[sbd:sed].sum()
                ts_table.set_value(index, 'si', si)

                # Cyclic fraction
                cf = si - spi
                ts_table.set_value(index, 'cf', cf)

                # Active fraction
                af = original.loc[sbd:max_date] - sp[:max_date]
                afi = af.sum()
                ts_table.set_value(index, 'afi', afi)

                # reference yr
                ref_yr = ts_table['sbd'].iloc[i]+((ts_table['sed'].iloc[i]-ts_table['sbd'].iloc[i])*2)/3
                ts_table.set_value(index, 'yr', ref_yr.year)

                # plt.ion()
                # original.plot(style='k:')
                # back.plot(style='c')
                # forward.plot(style='y')
                # sed_ts.plot(style='yo')
                # sbd_ts.plot(style='co')
                # plt.show()
            except (RuntimeError, Exception, ValueError):
                print('Error! populating ts_table went wrong, in year{1} @:{0}'.format(coord, index))
                continue

    except (RuntimeError, Exception, ValueError):
        print('Error! populating ts_table went wrong, in position:{0}'.format(coord))
        return

    plt.ion()
    # zoom the wresult window
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    # plt.figure(1)
    # ts_dek_unfix.plot(style='g--')
    # ts_dek.plot(style='r--')

    plt.figure(2)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.subplot(611)
    ts_clean.plot(style='r--')
    pks.plot(style='r--')
    pks.plot(style='ro')

    plt.subplot(612)
    # ts_dek_resc_unfix.plot(style='k:')
    ps.plot(style='g')
    ps.loc[ts_table.loc[:, 'sbd'].dropna()].plot(style='co')
    ps.loc[ts_table.loc[:, 'sed'].dropna()].plot(style='mo')

    # smoothed_crvs.plot(style='y.')
    plt.subplot(613)
    ts_table['sslp'].plot()

    plt.subplot(614)
    ts_table['sl'].plot()

    plt.subplot(615)
    ts_table['si'].plot()

    plt.show()
    plt.clf()

    return ts_table, err_table
