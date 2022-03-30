import pickle
from tqdm import tqdm
from urllib import request

def save_variable(v, filename):
  f = open(filename,'wb')
  pickle.dump(v, f)
  f.close()
  return filename
 
def load_variable(filename):
  f = open(filename,'rb')
  r = pickle.load(f)
  f.close()
  return r

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, desc):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=desc) as t:
        request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def print_search_results(clf):

    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_params_)
    print(
        "The best parameters are %s with a score of %0.2f"
        % (clf.best_params_, clf.best_score_)
    )
    print()
    print("Grid scores on development set:")
    print()
    means_train = clf.cv_results_["mean_train_score"]
    stds_train = clf.cv_results_["std_train_score"]
    means_test = clf.cv_results_["mean_test_score"]
    stds_test = clf.cv_results_["std_test_score"]
    for mean_train, std_train, mean_test, std_test, params in zip(means_train, 
    stds_train, means_test, stds_test, clf.cv_results_["params"]):
        print("train accuracy: %0.3f (+/-%0.03f), validation accuracy: %0.3f (+/-%0.03f) for %r" 
        % (mean_train, std_train * 2, mean_test, std_test * 2, params))

def plot_search_results(grid,p):
    """
    Params: 
        grid: A trained GridSearchCV object.
        p: The hyparameter
    """
    # Results from grid search
    results = grid.cv_results_
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']

    # convert param_grid from dictionary list to dictionary
    params=grid.param_grid[0]

    # Ploting results
    x = np.array(params[p])
    fig, ax = plt.subplots(figsize=(5, 2.7)) 
    ax.plot(x, means_train, '-', label='Train')  
    ax.plot(x, means_test, '-', label='Validation') 
    ax.fill_between(x, means_train - stds_train, means_train + stds_train, alpha=0.2)
    ax.fill_between(x, means_test - stds_test, means_test + stds_test, alpha=0.2)
    ax.set_xlabel(p)  
    ax.set_ylabel('Accuracy')
    # ax.set_xscale('log')  
    ax.legend()

def print_test_results(clf):
    print("Detailed classification report:")
    print()
    print("The model is trained on the size-210 train set.")
    print("The scores are computed on the size-70 test set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()