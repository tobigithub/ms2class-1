# import some library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import lightgbm as lgb
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

# Use large max_bin (may be slower)
# Use small learning_rate with large num_iterations
# Use large num_leaves (may cause over-fitting)
# Try dart

def gs():
    t = []
    warnings.filterwarnings('ignore')
    for i in [255]: # max_bin
        for l in [0.1]: # learning_late
            for m in [31, 100, 150]: # num_leaves
                for v in ['gbdt']: # boosting

                    # define and fit
                    gbm = lgb.LGBMClassifier(
                        objective='multiclass',
                        device = 'gpu',
                        max_bin=i,
                        learning_rate=l,
                        num_leaves=m,
                        boosting=v
                    )

                    gbm.fit(
                        X_train, 
                        y_train,
                        eval_set=[(X_test, y_test)],
                        early_stopping_rounds=50,
                        verbose=False
                    )

                    print(gbm.score(X_test, y_test))
                    print(
                        'max_bin: '+str(i),
                        'learning_late: '+str(l), 
                        'num_leaves: '+str(m), 
                        'boosting: '+str(v),
                    )
                    t.append([
                            gbm.score(X_test, y_test),
                            i, l, m, v
                    ])
                    
    t = pd.DataFrame(t, columns=[
        'Accuracy',
        'max_bin',
        'lr',
        'num_leaves',
        'boosting',
    ]).sort_values('Accuracy', ascending=False).to_csv('../result/LGBM_best.csv')

gs()
