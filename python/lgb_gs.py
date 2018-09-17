import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import lightgbm as lgb

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

gbm = lgb.LGBMClassifier(
    objective='multiclass',
    device = 'gpu',
#     eval_set=[(X_test, y_test)],
#     early_stopping_rounds=5
)

params = {
#     'learning_rate':[0.1],
    'num_leaves':[50, 100, 150],
    'max_depth':[5, 10, 15],
    'n_estimators':[10, 20, 40],
    'min_data_in_leaf':[10, 50, 100],
    'max_bin':[63, 127, 255],
    'boosting_type':['gbdt','dart']
#     'min_sum_hessian_in_leaf': [],
    
#     'metric': {'l2'},
#     'num_leaves': 5,
#     'learning_rate': 0.06,
#     'max_depth': 4,
#     'subsample': 0.95,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.85,
#     'bagging_freq': 4,
#     'min_data_in_leaf':4,
#     'min_sum_hessian_in_leaf': 0.8,
#     'verbose':10
}

grid_search = GridSearchCV(
    estimator = gbm, 
    param_grid = params, 
    cv = 3, 
    n_jobs = -1, 
    verbose = 1
)

grid_search.fit(X_train, y_train)

f = grid_search.best_estimator_
f.fit(X_train, y_train)
f.score(X_test, y_test)

loaded_model = pickle.load(open('../result/lgb_gs.pkl', 'rb'))
if f.score(X_test, y_test) > loaded_model.score(X_test, y_test):
    pickle.dump(f, open('../result/lgb_gs.pkl', "wb"))
