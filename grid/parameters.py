from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
"""
根据给定的超参数范围(序列),从中求解出较好的超参数值
本模块定义2个字典,用来指定各个分类(回归)模型需要计算的超参数范围
这两个字典内部又是有{model:param_dict}形式构成的
一般地,超参数范围指定地越多,消耗的计算时间越长,应该选取合适的超参数
此外,以字典形式提供,对于指向要搜索指定模型的超参数是放便的

"""
classification_grid_parameters = {
    # 指定合适的超参数序列(范围)
    SVC():  {
        'C': [0.0005, 0.001, 0.01, 0.1, 1, 10],
        'gamma' : [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    RandomForestClassifier():   {
        'n_estimators': [10, 40, 70, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1, 2],
        'max_features': [0.2, 0.5, 1, 2],
    },
    # 定义决策树参数空间
    DecisionTreeClassifier(): {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 5, 7, 10, 13],
        "min_samples_split": [1,2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    # GradientBoostingClassifier():   {
    #     'learning_rate': [0.05, 0.1, 0.3],
    #     'n_estimators': [40, 70, 100],
    #     'subsample': [0.3, 0.5, 0.7, 1],
    #     'min_samples_split': [0.2, 0.5, 0.7, 2],
    #     'min_samples_leaf': [0.2, 0.5, 1],
    #     'max_depth': [3, 7],
    #     'max_features': [1, 2, None],
    # },
    # KNeighborsClassifier(): {
    #     'weights': ['uniform', 'distance'],
    #     'p': [1, 2, 3, 4, 5],
    # },
    # MLPClassifier():    {
    #     'hidden_layer_sizes': [(200,), (300,), (400,), (128, 128), (256, 256)],
    #     'alpha': [0.001, 0.005, 0.01],
    #     'batch_size': [128, 256, 512, 1024],
    #     'learning_rate': ['constant', 'adaptive'],
    #     'max_iter': [200, 300, 400, 500]
    # },
    # BaggingClassifier():    {
    #     'n_estimators': [10, 30, 50, 60],
    #     'max_samples': [0.1, 0.3, 0.5, 0.8, 1.],
    #     'max_features': [0.2, 0.5, 1, 2],
    # }
}

regression_grid_parameters = {
    # SVR():  {
    #     'C': [0.0005, 0.001, 0.002, 0.01, 0.1, 1, 10],
    #     'gamma' : [0.001, 0.01, 0.1, 1],
    #     'kernel': ['rbf', 'poly', 'sigmoid']
    # },
    RandomForestRegressor():   {
        'n_estimators': [10, 40, 70, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1, 2],
        'max_features': [0.2, 0.5, 1, 2],
    },
    GradientBoostingRegressor():   {
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [40, 70, 100],
        'subsample': [0.3, 0.5, 0.7, 1],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1],
        'max_depth': [3, 7],
        'max_features': [1, 2, None],
    },
    KNeighborsRegressor(): {
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
    },
    MLPRegressor():    {
        'hidden_layer_sizes': [(200,), (200, 200), (300,), (400,)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [64, 128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300, 400, 500, 600, 700]
    },
    BaggingRegressor():    {
        'n_estimators': [10, 30, 50, 60],
        'max_samples': [0.1, 0.3, 0.5, 0.8, 1.],
        'max_features': [0.2, 0.5, 1, 2],
    }
}