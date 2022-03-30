import os
from collections import OrderedDict

import numpy as np
from numpy import pi as PI
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm, trange

import nfft
import schwimmbad

from const import curve_file, curve_file_2, meta_file, meta_file_2, orig_feature_file,\
    feature_file, LOG_FEATS, curve_url, meta_url
from utils import download_url


class LightCurve:
    def __init__(self, curve, meta):
        self.curve = curve
        self.meta = meta

def compute_nfft(sample_instants, sample_values):
    N = 1000
    T = sample_instants[-1] - sample_instants[0]
    sample_instants = (sample_instants - sample_instants[0]) / T - 0.5
    f_k_fast = nfft.nfft_adjoint(sample_instants, sample_values, N).real    # get real part?
    return (f_k_fast[0], f_k_fast[1])

def worker(tsobj):
    curve = tsobj.curve
    flux = curve.flux
    flux_err = curve.flux_err
    flux_err_ratio = flux / flux_err
    flux_err2 = flux * (flux_err ** 2)
    interval = np.diff(curve.mjd)

    feature = {'n_measure': len(curve),\
        'min_flux': flux.min(),\
        'max_flux': flux.max(),\
        'mean_flux': flux.mean(),\
        'med_flux': flux.median(),\
        'std_flux': flux.std(),\
        'skew_flux': skew(flux),\
        'min_flux_err': flux_err.min(),\
        'max_flux_err': flux_err.max(),\
        'mean_flux_err': flux_err.mean(),\
        'med_flux_err': flux_err.median(),\
        'std_flux_err': flux_err.std(),\
        'skew_flux_err': skew(flux_err),\
        'sum_flux_err_ratio': np.sum(flux_err_ratio),\
        'skew_flux_err_ratio': skew(flux_err_ratio),\
        'sum_flux_err2': np.sum(flux_err2),\
        'skew_flux_err2': skew(flux_err2),\
        'mean_interval': interval.mean(),\
        'max_interval': interval.max(),\
        }
    feature.update(tsobj.meta)

    pband = [0, 1, 2, 3, 4, 5]
    pbind = [(curve.passband == pb) for pb in pband]    # list of masks(ndarray)
    mjd = [curve[mask].mjd.to_numpy() for mask in pbind]     # list of time
    flux = [curve[mask].flux.to_numpy() for mask in pbind]   # list of flux
    for pb in pband:
        feature[f'fou1_{pb}'], feature[f'fou2_{pb}'] = compute_nfft(mjd[pb], flux[pb])
        feature[f'kur_{pb}'] = kurtosis(flux[pb])
        feature[f'skew_{pb}'] = skew(flux[pb])

    return feature


if __name__ == '__main__':
    if not os.path.exists(curve_file):
        # Download PLAsTiCC training set data
        download_url(curve_url, curve_file, 'Download PLAsTiCC training set data')

    
    if not os.path.exists(meta_file):
        # Download PLAsTiCC training set metadata
        download_url(meta_url, meta_file, 'Download PLAsTiCC training set metadata')
    
    if os.path.exists(meta_file_2) and os.path.exists(curve_file_2):
        metadata = pd.read_csv(meta_file_2)
        curve_data = pd.read_csv(curve_file_2)
    else:
        # Extract data with target values 42/90
        meta_dtype = {'object_id': np.int32, 'ra': np.float32, 'decl': np.float32,\
                'gal_l': np.float32, 'gal_b': np.float32, 'ddf': np.float32,\
                'hostgal_specz': np.float32, 'hostgal_photoz': np.float32,\
                'hostgal_photoz_err': np.float32, 'distmod': np.float32,\
                'mwebv': np.float32, 'target': np.int8}
        metadata = pd.read_csv(meta_file, dtype=meta_dtype)
        metadata = metadata[(metadata.target == 42) | (metadata.target == 90)]
        obj_id = metadata.object_id

        curve_dtype = {'object_id': np.int32, 'mjd': np.float64, 'passband': np.float32,\
                    'flux': np.float32, 'flux_err': np.float32, 'detected': np.bool8}
        curve_data = pd.read_csv(curve_file, dtype=curve_dtype)
        curve_data = curve_data[curve_data.object_id.isin(obj_id)]

        metadata.to_csv(meta_file_2, index=False)
        curve_data.to_csv(curve_file_2, index=False)


    n_obj = len(metadata)
    tsdict = OrderedDict()
    for i in trange(n_obj, desc='Recording Meta Features'):
        row = metadata.iloc[i]
        obj_id = row['object_id']
        ind = (curve_data['object_id'] == obj_id)
        curve = curve_data[ind]
        
        meta = {'hostgal_specz': row['hostgal_specz'],\
                'hostgal_photoz': row['hostgal_photoz'],\
                'hostgal_photoz_err': row['hostgal_photoz_err'],\
                'ra': row['ra'],\
                'decl': row['decl'],\
                'gal_l': row['gal_l'],\
                'gal_b': row['gal_b'],\
                'ddf': row['ddf'],\
                'distmod': row['distmod'],\
                'mwebv': row['mwebv'],\
                'target': row['target'] == 42
                }

        tsdict[obj_id] = LightCurve(curve, meta)

    del curve_data
    del metadata

    # Compute features
    if os.path.exists(orig_feature_file):
        feature_df = pd.read_csv(orig_feature_file)
    else:
        features_list = []
        with tqdm(total=n_obj, desc="Computing Features") as pbar:
            with schwimmbad.MultiPool() as pool:  
                results = pool.imap(worker, list(tsdict.values()))
                for feature in results:
                    features_list.append(feature)
                    pbar.update()

        feature_df = pd.DataFrame(features_list)
        feature_df.to_csv(orig_feature_file, index=False)

    # Remove target column     
    target = feature_df.pop('target')

    # Logscale transformation
    feature_df = feature_df.astype(np.float64).abs()
    print(feature_df[:10])

    feature_df[LOG_FEATS] = feature_df[LOG_FEATS].apply(np.log10)
    print(feature_df[:10])

    # Normalization/scaling and outliers
    feature_df = (feature_df - feature_df.quantile(.01)) / (feature_df.quantile(.99) - feature_df.quantile(.01)) 
    feature_df = feature_df * PI - PI / 2

    # Insert target to last column
    feature_df.insert(loc=feature_df.shape[1], column='target', value=target, allow_duplicates=False)
    print(feature_df[:10])

    # Save feature
    np.save(feature_file, feature_df.to_numpy())