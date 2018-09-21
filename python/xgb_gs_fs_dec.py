# import some library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import itertools
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# read data

df = pd.concat([
    pd.read_csv('../data/feature_selection_positive.csv', index_col=0),
    pd.read_csv('../data/decomp_pos.csv', index_col=0).drop('Subclass', axis=1)
], axis=1)

# divide objective and target
objective = df.Subclass
le = preprocessing.LabelEncoder()
objective = le.fit_transform(objective)
features = df.drop('Subclass', axis=1)

# train test split
random_state=np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    objective,
    test_size=0.2
)

# gridsearch

params = {
    'max_depth': [5, 7, 6],
    'min_child_weight': [1, 5, 3],
    'subsample': [0.3, 0.4, 0.5],
    'colsample_bytree':  [0.5, 0.6, 0.7]
}

xgb = XGBClassifier(
    device='gpu',
    gpu_id=1,
    updater='grow_gpu_hist',
    objective='multi:softmax',
    n_estimators=100
)

clf = GridSearchCV(
    xgb,
    params,
    verbose=2,
    cv=3,
    n_jobs=-1
)

clf.fit(X_train, y_train)
# pickle.dump(clf, open('../model/XGB_best_params_fs.sav', 'wb'))

t = clf.best_estimator_
pickle.dump(t, open('../model_gs/XGB_best_params.sav', 'wb'))

