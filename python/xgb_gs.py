import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import xgboost as xgb
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv('../data/feature_importance_pos.csv', index_col=0)
objective = df.Subclass
le = preprocessing.LabelEncoder()
objective = le.fit_transform(objective)
features = df.drop('Subclass', axis=1)

random_state=np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    objective,
    test_size=0.2
)

import warnings
warnings.filterwarnings('ignore')
param = {
    'tree_method':'gpu_hist',
    'gpu_id': 1
}

num_round = 100

t = xgb.XGBClassifier(**param)

params = {
    'objective':['multi:softmax'],
    'min_child_weight': [1, 2, 3],
#     'learning_rate':[0.1, 0.2, 0.3, 0.4], 
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'max_depth': [3, 5, 7],
#     'learning_rate':[0.1],
#     'n_estimators':[1000],
#     'max_delta_step':[5],
    'gamma':[0,3,10],
#     'colsample_bytree':[0.8],
#     'scale_pos_weight':[1],
}

grid_search = GridSearchCV(
    estimator = t, 
    param_grid = params, 
    cv = 3, 
    n_jobs = -1, 
    verbose = 1
)

start = time.time()
grid_search.fit(X_train, y_train)
elapsed_time = time.time() - start

f = grid_search.best_estimator_
f.fit(X_train, y_train)

f.score(X_test, y_test)

loaded_model = pickle.load(open('../result/xgb_gs.pkl', 'rb'))
if f.score(X_test, y_test) > loaded_model.score(X_test, y_test):
    pickle.dump(f, open('../result/xgb_gs.pkl', "wb"))
