import ml_collections as mlc
import sklearn
from sklearn.model_selection import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.neural_network import *

import autosklearn
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.ensembles import SingleBest
from sklearn.model_selection import *

from imblearn.over_sampling import SMOTE

from tdc import Evaluator

import functools
import time

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
        'svm': {
            'estimator': LinearSVC,
            'param_grid': {
                'penalty': ['l1', 'l2'],
                'loss': ['squared_hinge', 'hinge'],
                'C': [0.1, 1, 10, 100],
                'max_iter': [1000, 10000],
            },
            'scoring': 'roc_auc',
        },
        'k_nearest_neighbors': {
            'estimator': KNeighborsClassifier,
            'param_grid': {
                'n_neighbors': [1, 5, 9],
                'weights': ['uniform', 'distance'],
                'leaf_size': [10, 30, 50],
            },
            'scoring': 'roc_auc',
        },
        'mlp': {
            'estimator': MLPClassifier,
            'param_grid': {
                'hidden_layer_sizes': [(100, 100), (300, 300)],
                'activation': ['tanh', 'relu'],
                # 'learning_rate': ['invscaling', 'adaptive'],
                'solver': ['adam'],
                'max_iter': [100, 300],
            },
            'scoring': 'roc_auc',
        },
        'random_forest': {
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
        'svm': {
            'estimator': LinearSVR,
            'param_grid': {
                'epsilon': [0, 0.5, 1, 5, 10],
                'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                'C': [0.1, 1, 10, 100],
                'max_iter': [1000, 10000],
            },
            'scoring': 'r2',
        },
        'k_nearest_neighbors': {
            'estimator': KNeighborsRegressor,
            'param_grid': {
                'n_neighbors': [1, 5, 9],
                'weights': ['uniform', 'distance'],
                'leaf_size': [10, 30, 50],
            },
            'scoring': 'r2',
        },
        'mlp': {
            'estimator': MLPRegressor,
            'param_grid': {
                'hidden_layer_sizes': [(100, 100), (300, 300)],
                'activation': ['tanh', 'relu'],
                # 'learning_rate': ['invscaling', 'adaptive'],
                'solver': ['adam'],
                'max_iter': [100, 300],
            },
            'scoring': 'r2',
        },
        'random_forest': {
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
        verbose=3,
        n_jobs=1,
        seed=0,
        n_cv=5,
        balance=True,
    ):
        assert task_type in ['classification', 'regression']
        self.task_type = task_type

        if task_type == 'classification':
            self.config = AutoML_config['classification']
            self.model_types = list(AutoML_config['classification'].keys())
        elif task_type == 'regression':
            self.config = AutoML_config['regression']
            self.model_types = list(AutoML_config['regression'].keys())
        else:
            raise KeyError

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.seed = seed
        self.n_cv = n_cv
        self.balance = balance

        self.models = dict()

    def train(self, x_train, y_train, x_test, y_test, groups=None):
        self.init_data(x_train, y_train, x_test, y_test, groups)

        if groups is not None:
            # assert groups is not None  # using GroupKFold
            self.cv = GroupKFold(
                n_splits=self.n_cv,
            )
        else:
            self.cv = self.n_cv
        set_all_seed(self.seed)

        for i, model_type in enumerate(self.model_types):
            print(f'Training {model_type} ... [{i+1}/{len(self.model_types)}]')
            time_0 = time.time()
            self.train_single(model_type)
            print(f'Finished {model_type} - {time.time()-time_0:.1f}s')
        print('================== DONE ==================')

    def train_single(self, model_type):
        model_config = self.config[model_type]

        grid_search = GridSearchCV(
            functools.partial(model_config['estimator'], random_state=self.seed)()
            if model_type not in ['k_nearest_neighbors'] else model_config['estimator'](),
            param_grid=model_config['param_grid'],
            scoring=model_config['scoring'],
            cv=self.cv,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            refit=True,
        )
        grid_search.fit(self.x_train, self.y_train, groups=self.groups)

        self.models[model_type] = grid_search

    def predict(self, x_test):
        y_pred = dict()
        for model_type in self.models.keys():
            y_pred[model_type] = self.predict_single_model(x_test, model_type)
        return y_pred

    def predict_single_model(self, x_test, model_type):
        if isinstance(x_test, list):
            y_pred = []
            for x in x_test:
                y_pred.append(self.predict_single_data(x, model_type))
            y_pred = np.stack(y_pred, 0).mean(0)
        else:
            y_pred = self.predict_single_data(x_test, model_type)
        return y_pred

    def predict_single_data(self, x_test, model_type):
        y_pred = self.models[model_type].best_estimator_.predict(x_test)

        if self.task_type == 'regression':
            y_pred = self.descale_data(y_pred)

        return y_pred

    def init_data(self, x_train, y_train, x_test, y_test, groups):
        if isinstance(x_test, list):  # for n_noise_repeat
            # x_test = [x_test_1, x_test2, ...]
            self.x_test_list = x_test
            x_test = x_test[0]
        else:
            self.x_test_list = [x_test]

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

    def scale_data(self, y_train, y_test):
        self.scale_obj = StandardScaler()
        self.scale_obj.fit(y_train)
        return self.scale_obj.transform(y_train), self.scale_obj.transform(y_test)

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



class AutoSklearn(object):
    def __init__(
        self,
        task_type,
        time_left_for_this_task=3600,
        per_run_time_limit=360,
        n_jobs=16,
        tmp='./tmp',
        seed=42,
        memory_limit=1024*256,
        n_cv=5,
    ):
        # classification
        # adaboost bernoulli_nb decision_tree extra_trees gaussian_nb gradient_boosting k_nearest_neighbors
        # lda liblinear_svc libsvm_svc mlp multinomial_nb passive_aggressive qda random_forest sgd

        # regression
        # ['adaboost', 'ard_regression', 'decision_tree', 'extra_trees', 'gaussian_process', 'gradient_boosting',
        # 'k_nearest_neighbors', 'liblinear_svr', 'libsvm_svr', 'mlp', 'random_forest', 'sgd']
        assert task_type in ['classification', 'regression']
        self.task_type = task_type

        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.n_jobs = n_jobs
        self.tmp = tmp
        self.seed = seed
        self.memory_limit = memory_limit
        self.n_cv = n_cv

    def train(self, x_train, y_train, x_test, y_test, groups=None):
        self.init_data(x_train, y_train, x_test, y_test, groups)

        print(f'Begin autosklearn training ... ')
        time_0 = time.time()
        set_all_seed(self.seed)
        self.train_model()
        print(f'Finished autosklearn training - {time.time()-time_0:.1f}s')

    def init_data(self, x_train, y_train, x_test, y_test, groups=None):
        if isinstance(x_test, list):  # for n_noise_repeat
            # x_test = [x_test_1, x_test2, ...]
            self.x_test_list = x_test
            x_test = x_test[0]
        else:
            self.x_test_list = [x_test]

        y_train, y_test = self.pre_trans(y_train, y_test)

        if len(y_train.shape) == 2 and y_train.shape[-1] == 1:
            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.groups = groups

    def train_model(self):
        if self.groups is None:
            resampling_strategy = 'cv'
            resampling_strategy_arguments = None
            dataset_compression = True
            refit = False
        else:
            resampling_strategy = GroupKFold(n_splits=self.n_cv)
            resampling_strategy_arguments = dict(
                # train_size=0.7,
                # shuffle=True,
                # folds=5,
                groups=self.groups,
            )
            dataset_compression = False
            refit = True

        if self.task_type == 'classification':
            main_obj = autosklearn.classification.AutoSklearnClassifier
        elif self.task_type == 'regression':
            main_obj = autosklearn.regression.AutoSklearnRegressor
        else:
            raise KeyError

        automl = main_obj(
            tmp_folder=self.tmp,

            time_left_for_this_task=self.time_left_for_this_task,
            per_run_time_limit=self.per_run_time_limit,
            n_jobs=self.n_jobs,
            seed=self.seed,
            memory_limit=self.memory_limit,

            resampling_strategy=resampling_strategy,
            resampling_strategy_arguments=resampling_strategy_arguments,
            dataset_compression=dataset_compression,
        )
        automl.fit(self.x_train, self.y_train, X_test=self.x_test, y_test=self.y_test)

        if refit:
            automl.refit(self.x_train, self.y_train)

        self.model = automl

    def pre_trans(self, y_train, y_test):
        if self.task_type == 'regression':
            if (y_train < 0).all() and (y_train < 0).all():
                self.flip = True
                y_train, y_test = -y_train, -y_test
            else:
                self.flip = False

            # self.scaler = StandardScaler()  # MinMaxScaler()
            # self.scaler.fit(y_train)
            #
            # y_train = self.scaler.transform(y_train)
            # y_test = self.scaler.transform(y_test)

        return y_train, y_test

    def post_trans(self, y_pred):
        if self.task_type == 'regression':
            if self.flip:
                y_pred = - y_pred

        #     if len(y_pred.shape) == 1:
        #         y_pred = y_pred.reshape(-1, 1)
        #     y_pred = self.scaler.inverse_transform(y_pred)
        #     if len(y_pred.shape) == 2 and y_pred.shape[-1] == 1:
        #         y_pred = y_pred.reshape(-1)

        return y_pred

    def predict(self, x_test):
        if isinstance(x_test, list):
            y_pred = []
            for x in x_test:
                y_pred.append(self.predict_single(x))
            y_pred = np.stack(y_pred, 0).mean(0)
        else:
            y_pred = self.predict_single(x_test)
        return y_pred

    def predict_single(self, x_test):
        y_pred = self.model.predict(x_test)
        y_pred = self.post_trans(y_pred)

        return y_pred

    @staticmethod
    def calc_metric(y_true, y_pred, metric):
        evaluator = Evaluator(name=metric)
        score = evaluator(y_true, y_pred)
        return score






















