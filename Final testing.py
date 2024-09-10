from sklearnex import patch_sklearn
patch_sklearn()

import time
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

train_data = pd.read_csv("data/data/mitbih_train.csv", header=None)
X = train_data.drop(187, axis=1)
y = train_data[[187]]

test_data = pd.read_csv("data/data/mitbih_test.csv", header=None)
X1 = test_data.drop(187, axis=1)
y1 = test_data[[187]]

seed = 42

model0 = XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=seed, use_label_encoder=False)
model1 = KNeighborsClassifier(n_jobs=-1, n_neighbors=4, weights='distance')
model2 = RandomForestClassifier(n_estimators=600, n_jobs=-1, random_state=42, max_features='auto', max_depth=None)

classifiers = [('model0', model0), ('model1', model1), ('model2', model2)]

model3 = VotingClassifier(estimators=classifiers)

all_models = [model0, model1, model2, model3]

model_set = {0: "XGBClassifier",
             1: "KNeighborsClassifier",
             2: "RandomForestClassifier",
             3: "VotingClassifier"}

with open("Final Results.txt", 'w') as f:
    start_start = time.time()
    for i, m in enumerate(all_models):
        start = time.time()
        m.fit(X, y.to_numpy().reshape(-1, ))
        print("--" * 50)
        print(f"{model_set[i]}")
        print("--" * 50, file=f)
        print(f"{model_set[i]}", file=f)
        end = time.time()
        print('Time Taken', end - start, f"seconds for training {model_set[i]}")
        print("Testing on testing data:")
        print("Classification Report:")
        print(classification_report(y1, m.predict(X1)))
        print("--" * 50)
        print('Time Taken', end - start, f"seconds for training {model_set[i]}", file=f)
        print("Testing on testing data:", file=f)
        print("Classification Report:", file=f)
        print(classification_report(y1, m.predict(X1)), file=f)
        print("--" * 50, file=f)
    end_end = time.time()

    print("Total time taken:", end_end - start_start, "seconds.")
    print("Total time taken:", end_end - start_start, "seconds.", file=f)
