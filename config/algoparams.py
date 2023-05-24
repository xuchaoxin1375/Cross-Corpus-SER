from joblib import dump, load
from config.MetaPath import bclf
ava_cv_modes=("kfold","ss","sss")
# 在这里配置机器学习算法,他们将被渲染到GUI界面中(如果界面布局采用读取配置的方式)
#但是响应逻辑需要自行编写到事件处理循环中
#可以先到base模块中测试您的学习器,然后根据学习器的复杂程度调整(扩展)界面(例如集成学习)
#或者,不开放额外的参数,直接内置超参数(个体学习器组合)
ava_ML_algorithms = [
    "BEST_ML_MODEL",
    "SVC",
    "RandomForestClassifier",
    "MLPClassifier",
    "KNeighborsClassifier",
    "BaggingClassifier",
    "GradientBoostingClassifier",
    "DecisionTreeClassifier",
    "AdaBoostClassifier",
    "StackingClassifier"
]
ava_algorithms = [
    *ava_ML_algorithms,
    "RNN",
]
ava_svd_solver = ["auto", "full", "arpack", "randomized"]



##
def get_ML_bclf_dict():
    """返回各个`分类器`最优超参数和对应的参数下的模型字典
    注意Classifier而非Regressor

    Returns
    -------
    dict[str, estimator]
        分类器模型名字和sklearn模型对象构成的字典
    """
    bclf_estimators = load(bclf)
    for bclf_estimator in bclf_estimators:
        print(bclf_estimator)
    # audio_selected=get_example_audio_file()
    # None表示自动计算best_ML_model
    ML_estimators_dict = get_ML_best_estimators_dict(bclf_estimators)
    return ML_estimators_dict

def get_ML_best_estimators_dict(best_estimators):
    """best_estimators中各分类/回归模型中模型的名字和sklearn模型对象

    Parameters
    ----------
    bclf_estimators : tuple(estimator,parameters,score)
        
    Returns
    -------
    dict[str, estimator]
        模型名字和sklearn模型对象构成的字典
    """
    return {
        estimator.__class__.__name__: estimator for estimator, _, _ in best_estimators
    }

if __name__=="__main__":
    print(get_ML_bclf_dict())

random_state = 42