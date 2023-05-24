##
import pickle as pkl

from joblib import dump, load
from scipy.stats import randint
from sklearn.datasets import fetch_california_housing, load_iris, make_classification
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from utils import dump_pickle_by_name, load_pickle_by_name, now_utc_field_str


# Best for SVC: C=0.001, gamma=0.001, kernel='poly'
# Best for AdaBoostClassifier: {'algorithm': 'SAMME', 'learning_rate': 0.8, 'n_estimators': 60}
# Best for RandomForestClassifier: {'max_depth': 7, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 40}
# Best for GradientBoostingClassifier: {'learning_rate': 0.3, 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 70, 'subsample': 0.7}
# Best for DecisionTreeClassifier: {'criterion': 'entropy', 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
# Best for KNeighborsClassifier: {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
# Best for MLPClassifier: {'alpha': 0.005, 'batch_size': 256, 'hidden_layer_sizes': (300,), 'learning_rate': 'adaptive', 'max_iter': 500}


# from config.MetaPath import bclf,brgr


# bclf = load_pickle_by_name(best_clf)
# dump_pickle_by_name(bclf,"bclf.pkl")
# best_rgr = brgr
# bclf = load(best_clf)
# brgr = load(best_rgr)
##
# bag_best = (
#     BaggingClassifier(max_features=0.5, n_estimators=50),
#     {"max_features": 0.5, "max_samples": 1.0, "n_estimators": 50},
#     0.9210651450309082,
# )

# rf_best = (
#     RandomForestClassifier(max_depth=7, max_features=0.5, n_estimators=40),
#     {
#         "max_depth": 7,
#         "max_features": 0.5,
#         "min_samples_leaf": 1,
#         "min_samples_split": 2,
#         "n_estimators": 40,
#     },
#     0.8854018069424631,
# )

# gb_best = (
#     GradientBoostingClassifier(
#         learning_rate=0.3, loss="log_loss", max_depth=7, subsample=0.7
#     ),
#     {
#         "learning_rate": 0.3,
#         "max_depth": 7,
#         "max_features": None,
#         "min_samples_leaf": 1,
#         "min_samples_split": 2,
#         "n_estimators": 100,
#         "subsample": 0.7,
#     },
#     0.9476937708036139,
# )

# dt_best = (
#     DecisionTreeClassifier(
#         criterion="entropy",
#         max_depth=7,
#         max_features="sqrt",
#         min_samples_leaf=1,
#         min_samples_split=2,
#     ),
#     {
#         "criterion": "entropy",
#         "max_depth": 7,
#         "max_features": None,
#         "min_samples_leaf": 1,
#         "min_samples_split": 2,
#     },
#     "overfit",
# )
# ab_best = (
#     AdaBoostClassifier(n_estimators=60, learning_rate=0.8, algorithm="SAMME"),
#     {"algorithm": "SAMME", "learning_rate": 0.8, "n_estimators": 60},
#     "",
# )

# stacking
random_state = 1
estimators = [
    (
        "rf",
        RandomForestClassifier(n_estimators=10, max_depth=3, random_state=random_state),
    ),
    (
        "adab",
        AdaBoostClassifier(
            n_estimators=10, learning_rate=0.1, random_state=random_state
        ),
    ),
    # ("svr", make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))),
    # ("rdcv", RidgeClassifierCV()),
    # ("dt", DecisionTreeClassifier(random_state=random_state)),
]
# 简单堆叠
stack0 = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack0_best = (stack0, "", "")
##


bclf = load("bclf.joblib")
# 插入序列(单不实际保存修改)
bclf.append(stack0_best)
for item in bclf:
    print(item)

# 导出修改回模型
dump(bclf, "bclf.joblib")


