from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from sklearn.metrics import log_loss
import numpy as np
from xgboost import XGBClassifier


def load_matrix(validation = True):
    if validation:
        y_train = np.load('data/y_train_val.npy')
        X_train = np.load('data/X_train_val.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        return X_train.astype(np.float64), X_val.astype(np.float64), y_train.astype(np.float64), y_val.astype(np.float64)
    else:
        y_train = np.load('data/y_train.npy')
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        return X_train.astype(np.float64), X_test.astype(np.float64), y_train.astype(np.float64)

def objective(space):
    # model = XGBClassifier(**search_space, tree_method='gpu_hist', gpu_id=0, use_label_encoder=False, eval_metric='mlogloss')
    model = XGBClassifier(n_estimators = space['n_estimators'],
                            max_depth = int(space['max_depth']),
                            learning_rate = space['learning_rate'],
                            gamma = space['gamma'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree'],
                            tree_method='gpu_hist', gpu_id=0, use_label_encoder=False, eval_metric='mlogloss')
    X_train, X_val, y_train, y_val = load_matrix(validation = True)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)
    y_pred = y_pred[:,1].astype(np.float64)
    return {'loss': log_loss(y_val, y_pred), 'status': STATUS_OK}

# search_space = {
#     'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
#     'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
#     'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
#     'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
#     'subsample':        hp.uniform('subsample', 0.8, 1),
#     'n_estimators':     hp.choice('n_estimators', np.arange(100, 2000, 100)),
# }

search_space = {
    'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}

# implement Hyperopt
algorithm=tpe.suggest

best_params = fmin(
  fn=objective,
  space=search_space,
  algo=algorithm,
  max_evals=100)

print(best_params)