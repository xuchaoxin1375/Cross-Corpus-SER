"""
A script to grid search all parameters provided in parameters.py
including both classifiers and regressors.

Note that the execution of this script may take hours to search the 
best possible model parameters for various algorithms, feel free
to edit parameters.py on your need ( e.g remove some parameters for 
faster search )
"""

from joblib import dump,load

from emotion_recognition import EmotionRecognizer
from EF import e_config_def
from parameters import classification_grid_parameters, regression_grid_parameters

#配置识别模型在哪些情感上搜索最优超参数(情感越多,花费时间越长)
# emotion classes you want to perform grid search on
emotions_csv = e_config_def
# number of parallel jobs during the grid search
n_jobs = 4

#1.分类预测
# 构造用来保存各个模型最优超参数
best_estimators = []

for model, params_dict in classification_grid_parameters.items():
    mn=model.__class__.__name__
    if mn == "KNeighborsClassifier":
        # in case of a K-Nearest neighbors algorithm, set number of neighbors to the length of emotions
        params_dict['n_neighbors'] = [len(emotions_csv)]
    # 实例化情感分类器并读取数据
    er = EmotionRecognizer(model, emotions=emotions_csv)
    er.load_data()
    # 获取最优超参数
    best_estimator, best_params, cv_best_score = er.grid_search(params=params_dict, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions_csv} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation accuracy score!")

print(f"[I] saving best classifiers for {emotions_csv}...")

# 导出(保存)搜索结果
dump(best_estimators,f"grid/bclf.joblib")

# 2.回归预测

best_estimators = []

for model, params_dict in regression_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsRegressor":
        # in case of a K-Nearest neighbors algorithm
        # set number of neighbors to the length of emotions
        params_dict['n_neighbors'] = [len(emotions_csv)]
    er = EmotionRecognizer(model, emotions=emotions_csv, classification_task=False)
    er.load_data()
    best_estimator, best_params, cv_best_score = er.grid_search(params=params_dict, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions_csv} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation MAE score!")

print(f"[+] saving best regressors for {emotions_csv}...")
# pickle.dump(best_estimators, open(f"grid/best_regressors.pickle", "wb"))
dump(best_estimators, f"grid/brgr.joblib")




# Best for SVC: C=0.001, gamma=0.001, kernel='poly'
# Best for AdaBoostClassifier: {'algorithm': 'SAMME', 'learning_rate': 0.8, 'n_estimators': 60}
# Best for RandomForestClassifier: {'max_depth': 7, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 40}
# Best for GradientBoostingClassifier: {'learning_rate': 0.3, 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 70, 'subsample': 0.7}
# Best for DecisionTreeClassifier: {'criterion': 'entropy', 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
# Best for KNeighborsClassifier: {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
# Best for MLPClassifier: {'alpha': 0.005, 'batch_size': 256, 'hidden_layer_sizes': (300,), 'learning_rate': 'adaptive', 'max_iter': 500}