from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
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

def objective(search_space):
    model = RandomForestClassifier(**search_space, n_jobs=6, random_state=42)
    X_train, X_val, y_train, y_val = load_matrix(validation = True)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)
    y_pred = y_pred[:,1]
    return {'loss': log_loss(y_val, y_pred), 'status': STATUS_OK}
  
# new search space
search_space={'n_estimators':hp.randint('n_estimators',200,1000),
              
              'max_depth': hp.randint('max_depth',10,200),           
            
            'min_samples_split':hp.uniform('min_samples_split',0,1),   
             'min_samples_leaf':hp.randint('min_samples_leaf',1,10),
              
               'criterion':hp.choice('criterion',['gini','entropy']),
                
           'max_features':hp.choice('max_features',['sqrt', 'log2'])
             }
# {'criterion': 0, 'max_depth': 180, 'max_features': 1, 'min_samples_leaf': 3, 'min_samples_split': 0.0002042418947923803, 'n_estimators': 631}

# implement Hyperopt
algorithm=tpe.suggest

best_params = fmin(
  fn=objective,
  space=search_space,
  algo=algorithm,
  max_evals=200)

print(best_params)