from lightgbm import LGBMClassifier
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from pytest import param
from sklearn.metrics import log_loss
import numpy as np


def load_matrix(validation = True):
    if validation:
        y_train = np.load('data/y_train_val.npy')
        X_train = np.load('data/X_train_val.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        return X_train, X_val, y_train, y_val
    else:
        y_train = np.load('data/y_train.npy')
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        return X_train, X_test, y_train

def objective(params):
    # params = {
    #     'num_leaves': int(params['num_leaves']),
    #     'colsample_bytree': params['colsample_bytree'],
    #     'learning_rate': params['learning_rate'],
    #     'n_estimators': int(params['n_estimators']),
    #     'max_depth': int(params['max_depth']),
    #     'subsample': params['subsample'],
    #     'min_child_weight': params['min_child_weight'],
    #     'min_split_gain': params['min_split_gain'],
    #     'reg_alpha': params['reg_alpha'],
    #     'reg_lambda': params['reg_lambda']
    # }
    model = LGBMClassifier(**params, n_jobs=6)
    X_train, X_val, y_train, y_val = load_matrix(validation = True)
    mask = [11, 10, 23, 22]
    X_train, X_val = np.delete(X_train, mask, axis=1), np.delete(X_val, mask, axis=1)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)
    y_pred = y_pred[:,1]
    return {'loss': log_loss(y_val, y_pred), 'status': STATUS_OK}

# search_space = {
#     'learning_rate':    0.1,
#     'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
#     'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
#     'num_leaves':       hp.quniform('num_leaves', 8, 128, 2),
#     'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
#     'subsample':        hp.uniform('subsample', 0.8, 1),
#     'n_estimators':     500,
#     'min_split_gain':   hp.uniform('min_split_gain', 0, 1),
#     'reg_alpha':        hp.uniform('reg_alpha',0,1),
#     'reg_lambda':       hp.uniform('reg_lambda',0,1),
# }

search_space = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     hp.choice('n_estimators', np.arange(100, 2000, 100)),
}

# {'colsample_bytree': 0.4289902005468654, 'max_depth': 7, 'min_child_weight': 7,  'n_estimators': 500, 'learning_rate':0.1,
#  'min_split_gain': 0.8237097093914703, 'num_leaves': 78.0, 'reg_alpha': 0.4619815836877414, 'reg_lambda': 0.3702649278033648, 'subsample': 0.838062378418677}



# implement Hyperopt
algorithm=tpe.suggest

best_params = fmin(
  fn=objective,
  space=search_space,
  algo=algorithm,
  max_evals=200)

print(best_params)