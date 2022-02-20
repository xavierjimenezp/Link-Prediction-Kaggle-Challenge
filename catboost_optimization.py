from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from sklearn.metrics import log_loss
import numpy as np
from catboost import CatBoostClassifier

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

def get_catboost_params(space):
    params = dict()
    params['learning_rate'] = space['learning_rate']
    params['depth'] = int(space['depth'])
    params['l2_leaf_reg'] = space['l2_leaf_reg']
    params['border_count'] = space['border_count']
    #params['rsm'] = space['rsm']
    return params

def objective(space):
    params = get_catboost_params(space)
    
    model = CatBoostClassifier(iterations=10000,
                                learning_rate=params['learning_rate'],
                                depth=int(params['depth']),
                                loss_function='Logloss',
                                task_type="GPU",
                                eval_metric='Logloss',
                                l2_leaf_reg=params['l2_leaf_reg'],
                                early_stopping_rounds=3000,
                                od_type="Iter",
                                border_count=int(params['border_count']),
                                verbose=False
                                )
    X_train, X_val, y_train, y_val = load_matrix(validation = True)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)
    y_pred = y_pred[:,1].astype(np.float64)
    return {'loss': log_loss(y_val, y_pred), 'status': STATUS_OK}

search_space = {
        'depth': hp.quniform("depth", 1, 6, 1),
        'border_count': hp.uniform ('border_count', 32, 255),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 8),
       }

# implement Hyperopt
algorithm=tpe.suggest

best_params = fmin(
  fn=objective,
  space=search_space,
  algo=algorithm,
  trials = Trials(),
  max_evals=50)

print(best_params)