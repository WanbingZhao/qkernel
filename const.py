datadir = 'data'
figdir = 'fig'
meta_file = f'{datadir}/training_set_metadata.csv'
meta_file_2 = f'{datadir}/training_set_metadata_42_90.csv'
curve_file = f'{datadir}/training_set.csv.zip'
curve_file_2 = f'{datadir}/training_set_42_90.csv'

orig_feature_file = f'{datadir}/plasticc_feature.csv'
feature_file = f'{datadir}/plasticc_feature.npy'

LOG_FEATS = ['min_flux', 'max_flux', 'mean_flux', 'med_flux', 'std_flux', 'skew_flux',\
            'min_flux_err', 'max_flux_err', 'mean_flux_err', 'med_flux_err', 'std_flux_err',\
            'skew_flux_err', 'sum_flux_err_ratio', 'skew_flux_err_ratio', 'sum_flux_err2',\
            'skew_flux_err2', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err']

c1_opt_file = f'{datadir}/c1_opt.txt'
Cnl_opt_file = f'{datadir}/Cnl_opt.txt'
data280_file = f'{datadir}/data280.npy'
kernel280_file = f'{datadir}/kernel280.npy'
data1k_file = f'{datadir}/data1k.npy'
kernel1k_file = f'{datadir}/kernel1k.npy'

qubit_list = [5, 7, 10]

curve_url = 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/10384/120379/' \
            'compressed/training_set.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.' \
            'gserviceaccount.com&Expires=1649002492&Signature=Xuh8Ur%' \
            '2FGyOpVt214ReYp4lwaC2%2F2N9bRByunllQ2sFvBbxI5KorklHJK2hpmREG%2FPbWfMC8nvI94e3pR0uWHCEx20IUL%' \
            '2Bg30flUOii80Ar08RqYi0LInobNHs%2FyJujpRpMa7owi0gaOv%2BJhuIG8iJfa3tjpW1lszcwR72fJp%2BRXwk1oa67o' \
            'TWEKMV1zfmLUt9VtSXgWuFMdHsOaW4i96%2FgZejHlWNq0wiIV2z3H6%2FpjxI68Opquge2qcw9fvbhT3ciwHDFU8fUpl82' \
            'AU8U%2BCGjijsOhsOO4MDN7rXxk8c7Y5Mbtm3c7utOb9L%2Fcx4tVVKGNRYrj1owRmiYb4vH%2BVaQ%3D%3D&' \
            'response-content-disposition=attachment%3B+filename%3Dtraining_set.csv.zip'
meta_url = 'https://storage.googleapis.com/kagglesdsdata/competitions/10384/120379/training_set_metadata.' \
           'csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1649002498&Signature=' \
           'UvmuUhxIDuQbk3Cd4G0yGvan3GYODqRCLQvGkGuBkSDrqMtU3yB5Bfvrn%2B2KOxhUacIrVz4LzCXVsMwJDterD3MDte0ow' \
           'bZQQAPjXFEbtVK1fGRmI%2FE3IPdFGoAZsn6hAFLZmtVyJ0MkDqLtDzpDFSQhlIrycbKZCCzCH1Q1yyCFtVxSwC4cW5pE0Ud' \
           'HtmdFmH49D3QGMd2LvQl8uL2XTtsT0O%2BKrlrYxnxzjBNrfFHSVYo1FgJTKn%2FDUMWOCff6Sw0LG90Upyf0gJpVHhu9tfs' \
           'GpHpgUpk7JASi5zT2jQA%2Fv7vFv8UcokXxWZ5wCfCVjffeOTyGEWh20O9S4oj6IQ%3D%3D&response-content-disposition' \
           '=attachment%3B+filename%3Dtraining_set_metadata.csv'
