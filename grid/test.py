##
import pickle as pkl

from joblib import dump, load
from scipy.stats import randint
from sklearn.datasets import (fetch_california_housing, load_iris,
                              make_classification)
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from utils import dump_pickle_by_name, load_pickle_by_name




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
bag_best = (BaggingClassifier(max_features=0.5, n_estimators=50),
            {'max_features': 0.5, 'max_samples': 1.0, 'n_estimators': 50},
            0.9210651450309082)

rf_best=(RandomForestClassifier(max_depth=7, max_features=0.5, n_estimators=40), {'max_depth': 7, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 40}, 0.8854018069424631)

gb_best=(GradientBoostingClassifier(learning_rate=0.3, loss='log_loss', max_depth=7,
                           subsample=0.7), {'learning_rate': 0.3, 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.7}, 0.9476937708036139)
# bclf[1]=rf_best
# bclf[2]=gb_best
##
# dump(rf_best, "qq.joblib")
# bclf[-1] = bag_best
##
bclf_res=load("bclf.joblib")
for item in bclf_res:
    print(item)

# ##
# brgr
# ##
# # 修改回归模型
# bag_rgr_best = (BaggingRegressor(max_features=1, max_samples=0.1),
#                 {'max_features': 1, 'max_samples': 0.1, 'n_estimators': 10},
#                 0.6521001743540973)

                
# # brgr[-1]=bag_rgr_best

