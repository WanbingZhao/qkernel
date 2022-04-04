from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y

from QKernel import QKernel

# Wrapper class for the custom kernel qk.q_kernel_matrix
class Q_Kernel(BaseEstimator,TransformerMixin):
    def __init__(self, n_qubit=7, c1=1.0, noise_free=True, gamma=0.1, p=0.1):
        super(Q_Kernel,self).__init__()
        self.n_qubit = n_qubit
        self.c1 = c1
        self.noise_free=noise_free
        self.gamma = gamma
        self.p = p

    def transform(self, X):
        qk = QKernel(self.n_qubit, self.c1, self.noise_free, self.gamma, self.p)
        return qk.q_kernel_matrix(X, self.X_train_)

    def fit(self, X, y=None, **fit_params):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_train_ = X
        return self


def train(svm, train_data=0, n_qubits=7, noise_free = True, qgamma=0.1, qp=0.1,\
        cv=4, C=[1], c1=[1], rgamma=[0.012]):

    if svm == 'qdata':
        # Create a pipeline where our custom predefined kernel Q_Kernel
        # is run before SVC.
        pipe = Pipeline([
            ('qk', Q_Kernel()),
            ('svm', SVC()),
        ])
        # Set the parameter 'c1' of our custom kernel by
        # using the 'estimator__param' syntax.
        param_grid = dict([
            ('qk__c1', c1),
            ('qk__n_qubit', [n_qubits]),
            ('qk__noise_free', [noise_free]),
            ('qk__gamma', [qgamma]),
            ('qk__p', [qp]),
            ('svm__kernel', ['precomputed']),
            ('svm__C', C),
        ])
        # Do grid search to get the best parameter value of 'c1', 'C'.
        svm = GridSearchCV(pipe, param_grid, cv=cv, verbose=3, n_jobs=-1, return_train_score=True)
        X_train, y_train = train_data
        svm.fit(X_train, y_train)

    elif svm == 'rbf':
        param_grid = dict(gamma=rgamma, C=C)
        # Do grid search to get the best parameter value of 'gamma', 'C'.
        svm = GridSearchCV(SVC(), param_grid, cv=cv, verbose=3, n_jobs=-1, return_train_score=True)
        X_train, y_train = train_data
        svm.fit(X_train, y_train)

    elif svm == 'qkernel':
        param_grid = dict(kernel=['precomputed'], C=C)
        svm = GridSearchCV(SVC(), param_grid, cv=cv, verbose=3, n_jobs=-1, return_train_score=True)
        kernel, y_train = train_data
        svm.fit(kernel, y_train)
    
    return svm

