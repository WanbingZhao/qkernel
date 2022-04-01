# Set environment
import os
# os.system('pip install -r requirements.txt')

# Import library
import sys
import argparse
from time import time
from random import sample

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
import paddle
import matplotlib.pyplot as plt

from const import feature_file, c1_opt_file, Cnl_opt_file,\
    data280_file, kernel280_file, datadir, qubit_list, figdir,\
    data1k_file, kernel1k_file
from train import train
from utils import save_variable, load_variable, print_search_results, plot_search_results
from QKernel import QKernel

print(paddle.device.get_device())
paddle.set_device(paddle.device.get_device())
show_time = True

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='all', help='run which part of the code, all/tune_c1/' +
                                                           'dataset_size/data280/tune_Cnl/tune_Chw/qubit/NLvsHW')
parser.add_argument('--n_qubits', type=int, default=5, help='the number of qubits of register')
parser.add_argument('--n_samples', type=int, default=1000, help='the number of training samples')
parser.add_argument('--qgamma', type=float, default=0.1, help='noise parameter')
parser.add_argument('--qp', type=float, default=0.1, help='noise parameter')
parser.add_argument('--rgamma', type=float, default=0.012, help='gamma in rbf')
parser.add_argument('--cv', type=int, default=4, help='folds for cross validation')
args = parser.parse_args()

if not os.path.exists(datadir):
    os.mkdir(datadir)
if not os.path.exists(figdir):
    os.mkdir(figdir)

# Build dataset
if not os.path.exists(feature_file):
    # Preprocess
    os.system('python preprocess.py')
  

############### Start timing
t = time()

if os.path.exists(data1k_file):
    X_y = np.load(data1k_file, allow_pickle=True).astype(np.float64)
else:
    # Sample n_samples data
    feature = np.load(feature_file, allow_pickle=True).astype(np.float64)
    X_y = shuffle(feature, random_state=0, n_samples=args.n_samples)
    np.save(data1k_file, X_y)

X, y = np.split(X_y, [-1], 1)
y = np.squeeze(y)


############################
# Supplementary Figure 2a: Hyperparameter Tuning c1
# Tune c1 with args.n_sample training data

if args.run in ['all','tune_c1']:
    data = (X, y)
    c1_range = np.linspace(0, 1, 20)
    qsvm_c1 = train('qdata', data, n_qubits=args.n_qubits, cv=4, c1=c1_range)
    c1_opt = qsvm_c1.best_params_['c1']
    save_variable(c1_opt, c1_opt_file)

    # Plot c1 tuning result
    # can also use 'print_search_results()' to print result.
    plot_search_results(qsvm_c1, 'c1', 'tune_c1')

    if args.run == 'tune_c1':
        sys.exit(0)


# Load c1_opt
if args.run in ['dataset_size', 'data280', 'tune_Cnl', 'tune_Chw', 'NLvsHW']:
    if os.path.exists(c1_opt_file):
        c1_opt = load_variable(c1_opt_file)
    else:
        sys.exit('Tune c1 first.')

# Compute n_samples by n_samples kernel
if args.run in ['all', 'dataset_size', 'data280']:
    if os.path.exists(kernel1k_file):
        kernel = np.load(kernel1k_file, allow_pickle=True).astype(np.float64)
    else:
        qk = QKernel(args.n_qubits, c1_opt, noise_free=True)
        kernel = qk.q_kernel_matrix(X, X)
        np.save(kernel1k_file, kernel)


############################
# Figure 3: Learning Curve and Sample Variance

if args.run in ['all','dataset_size']:

    downsample_sizes = np.linspace(64, 375, 5).astype(int)

    qmean_train_scores = []
    qstd_train_scores = []
    qmean_test_scores = []
    qstd_test_scores = []

    rmean_train_scores = []
    rstd_train_scores = []
    rmean_test_scores = []
    rstd_test_scores = []

    for ds_size in downsample_sizes:
        print(f'Training on {ds_size} data')
        qmean_train_score, qstd_train_score, qmean_test_score, qstd_test_score = [0] * 4
        rmean_train_score, rstd_train_score, rmean_test_score, rstd_test_score = [0] * 4

        n_trial = 10
        for _ in range(n_trial):
            indices = sample(range(args.n_samples), ds_size)
            x_indices = np.tile(indices, (2, 1)).T
            y_indices = np.tile(indices, (2, 1))

            kernel_samples = kernel[x_indices, y_indices]
            X_samples, y_samples = X[indices], y[indices]

            qsvm = train('qkernel', (kernel_samples, y_samples), n_qubits=args.n_qubits, cv=4, c1=[c1_opt])
            qres = qsvm.cv_results_
            qmean_train_score += qres['mean_train_score']
            qstd_train_score += qres['std_train_score']
            qmean_test_score += qres['mean_test_score']
            qstd_test_score += qres['std_test_score']

            rsvm = train('rbf', (X_samples, y_samples), cv=4)
            rres = rsvm.cv_results_
            rmean_train_score += rres['mean_train_score']
            rstd_train_score += rres['std_train_score']
            rmean_test_score += rres['mean_test_score']
            rstd_test_score += rres['std_test_score']

        qmean_train_scores.append(qmean_train_score / n_trial)
        qstd_train_scores.append(qstd_train_score / n_trial)
        qmean_test_scores.append(qmean_test_score / n_trial)
        qstd_test_scores.append(qstd_test_score / n_trial)
        rmean_train_scores.append(rmean_train_score / n_trial)
        rstd_train_scores.append(rstd_train_score / n_trial)
        rmean_test_scores.append(rmean_test_score / n_trial)
        rstd_test_scores.append(rstd_test_score / n_trial)

    fig = plt.figure()
    plt.errorbar(downsample_sizes, qmean_train_scores, rstd_train_scores, linestyle='-', marker='o',\
                label='Noiseless train')
    plt.errorbar(downsample_sizes, qmean_test_scores, rstd_test_scores, linestyle='-', marker='o',\
                label='Noiseless validation')
    plt.errorbar(downsample_sizes, rmean_train_scores, rstd_train_scores, linestyle='--', marker='o',\
                label='RBF train')
    plt.errorbar(downsample_sizes, rmean_test_scores, rstd_test_scores, linestyle='--', marker='o',\
                label='RBF validation')
    plt.legend()
    plt.xlabel('Train fold size')
    plt.ylabel('Accuracy')
    plt.savefig(f'{figdir}/learning_curve.png')

    if args.run == 'dataset_size':
        sys.exit(0)


############################
# Build data280

if args.run in ['all', 'data280']:
    n_trial = 10
    n_splits = 4
    sample_size = 280
    indices_list = []
    acc_val = []
    for _ in n_trial:
        indices = sample(range(args.n_samples), sample_size)
        x_indices = np.tile(indices, (2, 1)).T
        y_indices = np.tile(indices, (2, 1))

        kernel280 = kernel[indices]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(kernel280):
            indices_list.append((indices, train_index, test_index))
            kernel_train, kernel_test = kernel280[train_index], kernel280[test_index]
            y_train, y_test = y[train_index], y[test_index]
            qsvm280 = train('qkernel', (kernel_train, y_train), n_qubits=args.n_qubits, c1 = c1_opt)
            y_test_pred = qsvm280.predict(kernel_test)
            acc_val = acc_val.append(np.array(y_test_pred == y_test, dtype=int).sum() / len(y_test))

    acc_val = np.asarray(acc_val)
    min_idx = (np.abs(acc_val - np.mean(acc_val))).argmin()

    indices, train_idx, test_idx = indices_list[min_idx]
    X_y280 = X_y[indices]
    data280 = np.vstack(X_y280[train_idx], X_y280[test_idx])
    np.save(data280_file, data280)

# Load data280 and kernel280
if args.run in ['tune_Cnl', 'tune_Chw']:
    if os.path.exists(data280_file):
        data280 = np.load(data280_file, allow_pickle=True).astype(np.float64)
        X280, y280 = np.split(data280, [-1], 1)
        y280 = np.squeeze(y280)
    else:
        sys.exit('Data280 is required.')
if args.run == 'tune_Cnl':
    if os.path.exists(kernel280_file):
        kernel280 = np.load(kernel280_file, allow_pickle=True).astype(np.float64)
    else:
        sys.exit('Kernel280 is required.')


############################
# Supplementary Figure 2b: Hyperparameter Tuning C
# Tune C with size-210 training data on noiseless circuit

if args.run in ['all', 'tune_Cnl']:
    data = (kernel280[:210], y280[:210])
    Cnl_range = 10.0 ** np.linspace(-1, 1, 20)
    qsvm_Cnl = train('qkernel', data, n_qubits=args.n_qubits, cv=4, C=[Cnl_range])
    Cnl_opt = qsvm_Cnl.best_params_['C']
    save_variable(Cnl_opt, Cnl_opt_file)

    # Plot C_noiseless tuning result
    # can also use 'print_search_results()' to print result.
    plot_search_results(qsvm_Cnl, 'C', 'tune_Cnl')

if args.run == 'NLvsHW':
    if os.path.exists(Cnl_opt_file):
        load_variable(Cnl_opt_file)
    else:
        sys.exit('Tune C_noiseless first.')


############################
# Supplementary Figure 8: Hyperparameter Tuning C
# Tune C with size-210 training data on hardware with noise

if args.run in ['all', 'tune_Chw', 'NLvsHW']:
    acc_train_hw = []
    acc_test_hw = []
    X_train, X_test, y_train, y_test = train_test_split(X280, y280, train_size=210, shuffle=False)
    data = (X_train, y_train)

    for n_qubits in qubit_list:
        Chw_range = np.linspace(0.4, 1.0, 20)
        qsvm_Chw= train('qdata', data, n_qubits=n_qubits, cv=4, C=[Chw_range], c1=c1_opt)

        # Plot C_noise tuning result
        # can also use 'print_search_results()' to print result.
        plot_search_results(qsvm_Chw, 'C', f'tune_Chw_{n_qubits}')
        y_train_pred = qsvm_Chw.predict(X_train)
        y_test_pred = qsvm_Chw.predict(X_test)
        acc_train_hw.append(np.array(y_train_pred == y_train, dtype=int).sum() / len(y_train))
        acc_test_hw.append(np.array(y_test_pred  == y_test, dtype=int).sum() / len(y_test))


############################
# Figure 4c: Noiseless vs. Experimental Results

if args.run in ['all', 'NLvsHW']:
    mean_acc_train_nl = []
    std_acc_train_nl = []
    mean_acc_test_nl = []
    std_acc_test_nl = []

    for n_qubits in qubit_list:
        # Compute n_samples by n_samples kernel
        qk = QKernel(n_qubits, c1_opt, noise_free=True)
        kernel = qk.q_kernel_matrix(X, X)
        n_trial = 10
        train_acc = []
        test_acc = []

        for _ in range(n_trial):
            sample_size = 280
            train_size = 210

            indices = sample(range(args.n_samples), sample_size)
            x_indices = np.tile(indices, (2, 1)).T
            y_indices = np.tile(indices, (2, 1))

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)

            kernel_samples = kernel[x_indices, y_indices]
            train_kernel = kernel_samples[:train_size, :train_size]

            data = (train_kernel, np.squeeze(y))
            qsvm_Cnl = train('qkernel', data, c1=c1_opt, C=Cnl_opt)

            y_train_pred = qsvm_Cnl.predict(X_train)
            y_test_pred = qsvm_Cnl.predict(X_test)

            train_acc.append(np.array(y_train_pred == y_train, dtype=int).sum() / len(y_train))
            test_acc.append(np.array(y_test_pred == y_test, dtype=int).sum() / len(y_test))

        mean_acc_train_nl.append(np.mean(train_acc))
        std_acc_train_nl.append(np.std(train_acc))
        mean_acc_test_nl.append(np.mean(test_acc))
        std_acc_test_nl.append(np.std(test_acc))


    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(5, 2.7))
    l1 = axs[0].errorbar(qubit_list, mean_acc_train_nl, std_acc_train_nl, marker='o')
    l2 = axs[0].errorbar(qubit_list, mean_acc_train_nl, std_acc_train_nl, marker='s')
    axs[0].set_title('Noiseless')
    l3 = axs[1].errorbar(qubit_list, acc_train_hw, 0, marker='o')
    l4 =  axs[1].errorbar(qubit_list, acc_test_hw, 0, marker='s')
    axs[1].set_title('Experiment')

    plt.xlabel("Number of qubits")
    plt.ylabel("Accuracy")

    # Create the legend
    fig.legend([l1, l2], labels=['Train', 'Test'])
    plt.savefig(f'{figdir}/NLvsHW.png')

    ############### Show elapsed time
    if show_time:
        print('time used: %d s' % (time() - t))




