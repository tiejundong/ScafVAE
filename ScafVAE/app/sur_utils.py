import functools
import time

from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.metrics import roc_curve

from imblearn.over_sampling import SMOTE

from tdc import Evaluator

from ScafVAE.utils.common import *


AutoML_config = {
    'classification': {
        'adaboost': {
            'estimator': AdaBoostClassifier,
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1],
                'algorithm': ['SAMME', 'SAMME.R']
            },
            'scoring': 'roc_auc',
        },
        'SVM': {
            'estimator': functools.partial(SVC, probability=True),
            'param_grid': {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10],
            },
            'scoring': 'roc_auc',
        },
        'KNN': {
            'estimator': KNeighborsClassifier,
            'param_grid': {
                'n_neighbors': [1, 5, 9],
                'weights': ['uniform', 'distance'],
                'leaf_size': [10, 30, 50],
            },
            'scoring': 'roc_auc',
        },
        'MLP': {
            'estimator': MLPClassifier,
            'param_grid': {
                'hidden_layer_sizes': [(100, 100), (200, 200)],
                'activation': ['tanh', 'relu'],
                # 'learning_rate': ['invscaling', 'adaptive'],
                'solver': ['adam'],
                'max_iter': [100, 200],
            },
            'scoring': 'roc_auc',
        },
        'RF': {
            'estimator': RandomForestClassifier,
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 40],
                'max_features': ['auto', 'sqrt'],
                'criterion': ['gini', 'entropy'],
            },
            'scoring': 'roc_auc',
        },
    },
    'regression': {
        'adaboost': {
            'estimator': AdaBoostRegressor,
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1],
                'loss': ['linear', 'square']
            },
            'scoring': 'r2',
        },
        'SVM': {
            'estimator': SVR,
            'param_grid': {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10],
            },
            'scoring': 'r2',
        },
        'KNN': {
            'estimator': KNeighborsRegressor,
            'param_grid': {
                'n_neighbors': [1, 5, 9],
                'weights': ['uniform', 'distance'],
                'leaf_size': [10, 30, 50],
            },
            'scoring': 'r2',
        },
        'MLP': {
            'estimator': MLPRegressor,
            'param_grid': {
                'hidden_layer_sizes': [(100, 100), (200, 200)],
                'activation': ['tanh', 'relu'],
                # 'learning_rate': ['invscaling', 'adaptive'],
                'solver': ['adam'],
                'max_iter': [100, 200],
            },
            'scoring': 'r2',
        },
        'RF': {
            'estimator': RandomForestRegressor,
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 40],
                'max_features': ['auto', 'sqrt'],
                'criterion': ['squared_error', 'friedman_mse', 'poisson', 'absolute_error'],
            },
            'scoring': 'r2',
        },
    },
}


class AutoML(object):
    def __init__(
        self,
        task_type,
        model_type,
        verbose=3,
        n_jobs=1,
        seed=0,
        n_cv=5,
        balance=True,
    ):
        assert task_type in ['classification', 'regression']
        self.task_type = task_type

        if isinstance(model_type, str):
            assert model_type in ['adaboost', 'SVM', 'KNN', 'MLP', 'RF']
            self.model_is_custom = False
            self.model_type = model_type

            if task_type == 'classification':
                self.model_config = AutoML_config['classification'][model_type]
            elif task_type == 'regression':
                self.model_config = AutoML_config['regression'][model_type]
        else:
            self.model_is_custom = True
            self.model_type = model_type
            self.model_config = None

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.seed = seed
        self.n_cv = n_cv
        self.balance = balance

        self.models = dict()

    def train(self, x_train, y_train, x_test, y_test, groups=None):
        self.init_data(x_train, y_train, x_test, y_test, groups)
        if groups is not None:
            self.cv = GroupKFold(n_splits=self.n_cv)
        else:
            self.cv = self.n_cv
        set_all_seed(self.seed)

        time_0 = time.time()

        if self.model_is_custom:
            self.train_custom()
        else:
            self.train_grid()

        print(f'Finished training - {time.time()-time_0:.1f}s')

    def train_grid(self):
        model_config = self.model_config

        grid_search = GridSearchCV(
            functools.partial(model_config['estimator'], random_state=self.seed)()
            if self.model_type not in ['KNN'] else model_config['estimator'](),
            param_grid=model_config['param_grid'],
            scoring=model_config['scoring'],
            cv=self.cv,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            refit=True,
        )
        grid_search.fit(self.x_train, self.y_train, groups=self.groups)

        self.trained_model = grid_search.best_estimator_

    def train_custom(self):
        self.model_type.fit(self.x_train, self.y_train)
        self.trained_model = self.model_type

    def init_data(self, x_train, y_train, x_test, y_test, groups):
        if self.balance and self.task_type == 'classification':
            x_train, y_train, groups = self.balance_data(x_train, y_train, groups=groups)

        if self.task_type == 'regression':
            y_train, y_test = self.scale_data(y_train, y_test)

        if len(y_train.shape) == 2 and y_train.shape[-1] == 1:
            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.groups = groups

    def balance_data(self, x_train, y_train, groups=None):
        smote = SMOTE(random_state=self.seed)

        idx = np.arange(x_train.shape[0]).reshape(-1, 1)
        idx_resampled, y_train = smote.fit_resample(idx, y_train)
        idx_resampled = idx_resampled.reshape(-1)

        x_train = x_train[idx_resampled]

        if groups is not None:
            groups = groups[idx_resampled]

        return x_train, y_train, groups

    def predict(self, x_test):
        if self.task_type == 'regression':
            y_pred = self.trained_model.predict(x_test)
            y_pred = self.descale_data(y_pred)
        else:
            y_pred = self.trained_model.predict_proba(x_test)[:, 1].reshape(-1)

        return y_pred

    def scale_data(self, y_train, y_test):
        self.scale_obj = StandardScaler()
        self.scale_obj.fit(y_train.reshape(-1, 1))
        return self.scale_obj.transform(y_train.reshape(-1, 1)), self.scale_obj.transform(y_test.reshape(-1, 1))

    def descale_data(self, y_pred):
        y_pred = y_pred.reshape(-1, 1)
        y_pred = self.scale_obj.inverse_transform(y_pred)
        y_pred = y_pred.reshape(-1)
        return y_pred

    @staticmethod
    def calc_metric(y_true, y_pred, metric):
        evaluator = Evaluator(name=metric)
        score = evaluator(y_true, y_pred)
        return score






















