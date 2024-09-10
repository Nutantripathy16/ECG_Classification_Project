from sklearnex import patch_sklearn
patch_sklearn()

import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("data/data/mitbih_train.csv", header=None)
X = data.drop(187, axis=1)
y = data[[187]]

seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed, stratify=y)

model0 = XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=seed, use_label_encoder=False,
                       sampling_method='gradient_based')
model1 = KNeighborsClassifier(n_jobs=-1)
model2 = RandomForestClassifier(n_jobs=-1, random_state=seed)

classifiers = [('model0', model0), ('model1', model1), ('model2', model2)]

all_models = [model0, model1, model2]

model_set = {0: "XGBClassifier",
             1: "KNeighborsClassifier",
             2: "RandomForestClassifier"}

param_grid_XGB = {'learning_rate': np.geomspace(0.000001, 0.01, 10), 'max_depth': np.linspace(1, 10, 10, dtype=int),
                  'subsample': [.3, .5, .7], 'grow_policy': ['depthwise', 'lossguide']}
param_grid_KNC = {'n_neighbors': np.linspace(1, 10, 10, dtype=int), 'weights': ['uniform', 'distance']}
param_grid_RFC = {'n_estimators': np.linspace(100, 1000, 10, dtype=int), 'max_depth': [4, 5, None],
                  'max_features': ['auto', 'sqrt', 'log2']}

parameters = [param_grid_XGB, param_grid_KNC, param_grid_RFC]

with open("Hyper Parameter Tuning.txt", 'w') as f:
    start_start = time.time()
    for (i, m), param_grid in zip(enumerate(all_models), parameters):
        start = time.time()
        grid_search = GridSearchCV(m, param_grid, n_jobs=-1, cv=5)
        grid_search.fit(X_train, y_train.to_numpy().reshape(-1,))
        print("--" * 50)
        print(f"{model_set[i]}")
        print("Classification Report")
        print(classification_report(y_test, grid_search.predict(X_test)))
        print("Best Model Found:")
        print(grid_search.best_estimator_)
        print("Best Parameters:")
        print(grid_search.best_params_)
        end = time.time()
        print('Time Taken', end - start, f"seconds for {model_set[i]} GridSearchCV")
        print("--" * 50)
        print("--" * 50, file=f)
        print(f"{model_set[i]}", file=f)
        print("Classification Report", file=f)
        print(classification_report(y_test, grid_search.predict(X_test)), file=f)
        print("Best Model Found:", file=f)
        print(grid_search.best_estimator_, file=f)
        print("Best Parameters:", file=f)
        print(grid_search.best_params_, file=f)
        print('Time Taken', end - start, f"seconds for {model_set[i]} GridSearchCV", file=f)
        print("--" * 50, file=f)
    end_end = time.time()

    print("Total time taken for hyper parameter tuning:", end_end - start_start, "seconds.")
    print("Total time taken for hyper parameter tuning:", end_end - start_start, "seconds.", file=f)
