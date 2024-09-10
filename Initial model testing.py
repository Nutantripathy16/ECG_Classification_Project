from sklearnex import patch_sklearn
patch_sklearn()

import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

data = pd.read_csv("data/data/mitbih_train.csv", header=None)
N = 20000
new = data.groupby(187, group_keys=False).apply(lambda x: x.sample(int(np.rint(N * len(x) / len(data))))).sample(
    frac=1).reset_index(drop=True)
X = new.drop(187, axis=1)
y = new[[187]]

seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed, stratify=y)

model0 = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=seed)
model1 = SVC(random_state=seed)
model2 = KNeighborsClassifier(n_neighbors=5)
model3 = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=seed)
model4 = RandomForestClassifier(n_jobs=-1, random_state=seed, n_estimators=500)
model5 = GradientBoostingClassifier(n_estimators=500, learning_rate=.01, random_state=seed)
model6 = GradientBoostingClassifier(n_estimators=500, learning_rate=.01, random_state=seed, subsample=.8,
                                    max_features=.2)
model7 = DecisionTreeClassifier(random_state=seed)
model8 = LogisticRegression(n_jobs=-1, random_state=seed, max_iter=500)
model9 = OneVsRestClassifier(model7)
model10 = BaggingClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
model11 = GaussianNB()
model12 = LinearDiscriminantAnalysis()
model13 = QuadraticDiscriminantAnalysis()


all_models = [model0, model1, model2, model3, model4, model5, model6,
              model7, model8, model9, model10, model11, model12, model13]

model_set = {0: "XGBClassifier(tree_method='gpu_hist', gpu_id=0, random_state=seed)",
             1: "SVC(random_state=seed)",
             2: "KNeighborsClassifier(n_neighbors=5)",
             3: "AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=seed)",
             4: "RandomForestClassifier(n_jobs=-1, random_state=seed, n_estimators=500)",
             5: "GradientBoostingClassifier(n_estimators=500, learning_rate=.01, random_state=seed)",
             6: "GradientBoostingClassifier(n_estimators=500, learning_rate=.01, random_state=seed, subsample=.8, max_features=.2)",
             7: "DecisionTreeClassifier(random_state=seed)",
             8: "LogisticRegression(n_jobs=-1, random_state=seed, max_iter=500)",
             9: "OneVsRestClassifier(model7)",
             10: "BaggingClassifier(n_estimators=500, n_jobs=-1, random_state=seed)",
             11: "GaussianNB()",
             12: "LinearDiscriminantAnalysis()",
             13: "QuadraticDiscriminantAnalysis()"}

with open("First results.txt", 'w') as f:
    start_start = time.time()
    for i, m in enumerate(all_models):
        start = time.time()
        m.fit(X_train, y_train)
        print("--" * 50)
        print(f"{model_set[i]}")
        print("Classification Report")
        print(classification_report(y_test, m.predict(X_test)))
        print("--" * 50, file=f)
        print(f"{model_set[i]}", file=f)
        print("Classification Report", file=f)
        print(classification_report(y_test, m.predict(X_test)), file=f)
        end = time.time()
        print('Time Taken', end - start, f"seconds for {model_set[i]}")
        print("--" * 50)
        print('Time Taken', end - start, f"seconds for {model_set[i]}", file=f)
        print("--" * 50, file=f)
    end_end = time.time()

    print("Total time taken for model testing:", end_end - start_start, "seconds.")
    print("Total time taken for model testing:", end_end - start_start, "seconds.", file=f)
