datadir = 'data'
figdir = 'fig'
meta_file = f'{datadir}/training_set_metadata.csv'
meta_file_2 = f'{datadir}/training_set_metadata_42_90.csv'
curve_file = f'{datadir}/training_set.csv.zip'
curve_file_2 = f'{datadir}/training_set_42_90.csv'

orig_feature_file = f'{datadir}/plasticc_feature.csv'
feature_file = f'{datadir}/plasticc_feature.npy'
label_file = f'{datadir}/plasticc_label.npy'

LOG_FEATS = ['min_flux', 'max_flux', 'mean_flux', 'med_flux', 'std_flux', 'skew_flux',\
            'min_flux_err', 'max_flux_err', 'mean_flux_err', 'med_flux_err', 'std_flux_err',\
            'skew_flux_err', 'sum_flux_err_ratio', 'skew_flux_err_ratio', 'sum_flux_err2',\
            'skew_flux_err2', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err']

c1_opt_file = f'{datadir}/c1_opt.txt'
Cnl_opt_file = f'{datadir}/Cnl_opt.txt'
data280_file = f'{datadir}/data280.npy'
kernel280_file = f'{datadir}/kernel280.npy'

qubit_list = [5, 7, 10]

curve_url = 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/10384/120379/compressed/training_set.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1648717257&Signature=i%2BjTWlsttVzLFJk4pvrqlPyL7PVtb1vbE3mhFJrKplHDel6qLKfEgnYRt6vOxL%2BbVCeyflaMKcnVS2Ri9Pr5UMbJdABCHR%2B8ng1P3vPbYLBrVxl%2FvLqGnDG%2FxdovMGDtZCMY3VV6PQ4WqLSmEGJKCTmPW%2Byb71EvmQvMNDSPNlevjWcnbpc0g7qqkRwNuvlC7Woz0uyhixrHGJSJo0q1gpI0RUsTI2k8R7b5C7HljcTbIxJh4fauIVFJIXgSai7NXjjynZZJ6146PTvpc6IkyhILQBWD%2FOC06K70Im7TJK2jsDPiREHwC3FB1Y%2BRwjve%2ByT9DE6yX4%2F%2B4iEqCyifwg%3D%3D&response-content-disposition=attachment%3B+filename%3Dtraining_set.csv.zip'
meta_url = 'https://storage.googleapis.com/kagglesdsdata/competitions/10384/120379/training_set_metadata.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1648717315&Signature=mTfMJnh11RplubEL3duX8G9P8BEcgMRRiKbT%2FVh3lluDbLrFgUNEt976uDrPnUyxSj0%2FOemLSPCtnUxO7uaIcqrVRScPQyzhxCLJZ41i9OhVuS%2FlyoLXJkBu%2FNjmGDobBzLpbdBlfVJ29c0AP0hVTl%2BK5AptXlrQk%2BhvIHpnTL28mBmLwpp%2Ftt70cIptaUVZL6Jqv4gbBJuZjvFW0KxNH8HkMiHwGeWAFx%2FT%2FUaXYlnODBrKhJRUfzLttsLh6%2FgTvffhFb1%2BuaT6uAx%2BSSii2CazpzaM50zHo%2BaOqEeQvSZ6b3TEqbmkYDvlk39JT8X7ekJNuG2Rt0nisq430L91tA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtraining_set_metadata.csv'
