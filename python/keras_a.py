# import some library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools
import time
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from keras import models
from keras import layers
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Dense, GaussianNoise, GaussianDropout
from keras.models import Sequential, Model
from keras.regularizers import l2, l1
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras.callbacks import LearningRateScheduler
from keras.metrics import categorical_accuracy
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1, l2
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1",
        allow_growth=True,
        #         per_process_gpu_memory_fraction=0.5
    )
)
set_session(tf.Session(config=config))

# read data
df = pd.read_csv('../data/feature_importance_pos.csv', index_col=0)

# divide objective and target
objective = df.Subclass
le = preprocessing.LabelEncoder()
objective = le.fit_transform(objective)
features = df.drop('Subclass', axis=1)

# train test split
random_state = np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    features,
    objective,
    test_size=0.2
)

# transform  for keras's target label
y_train_for_keras = np_utils.to_categorical(y_train)
y_test_for_keras = np_utils.to_categorical(y_test)

# gridsearch
for i in [128, 256, 512]:  # layer
    for p in ['SGD', 'RMSprop', 'Adagrad']:  # optimizers
        for m in ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']:  # activations
            tf.set_random_seed(42)

            # make keras model
            start = time.time()
            inputs = Input(shape=(X_train.shape[1],))
            x = Dense(i, activation=m)(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(i, activation=m)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(i, activation=m)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(i, activation=m)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            predictions = Dense(
                len(df['Subclass'].value_counts()), activation='softmax')(x)
            model = Model(inputs=inputs, outputs=predictions)

            # compile
            model.compile(
                loss='categorical_crossentropy',
                optimizer=p,
                metrics=['accuracy']
            )

            epochs = 100
            batch_size = 1000
            es = EarlyStopping(monitor='val_loss', patience=20)

            history = model.fit(
                X_train,
                y_train_for_keras,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test_for_keras),
                verbose=0,
                callbacks=[
                    es,
                ]
            )

            print(i, p, model.evaluate(X_test, y_test_for_keras, verbose=0)[1])

            if model.evaluate(X_test, y_test_for_keras, verbose=0)[1] > load_model('../model/Keras_fs_pos.h5').evaluate(X_test, y_test_for_keras, verbose=0)[1]:
                model.save('../model/Keras_fs_pos.h5')

