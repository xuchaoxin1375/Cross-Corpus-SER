##
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split

from audio.core import best_estimators

##
# 生成示例数据集
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

##
# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# 在训练集上拟合模型
clf.fit(X_train, y_train)
print(type(clf))

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 输出分类器的准确率
print("Accuracy:", clf.score(X_test, y_test))
## 
#直接使用交叉验证

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
data=cross_validate(clf,X_train, y_train)
# print(res)
for item in data.items():
    print(item)
##

from data_extractor import *
from config.MetaPath import *

pair_dict=select_meta_dict(pair2)
data=load_data_from_meta(**pair_dict)
X_train=data["X_train"]
y_train=data["y_train"]
X_test=data["X_test"]
y_test=data["y_test"]
# from sklearn.svm import SVC
# svc = SVC(kernel='linear')
# clf=svc
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
##
bes=best_estimators()
estimators= [tp[0] for tp in bes]
##
# for estimator in estimators:
#     estimator.fit(X_train, y_train)
#     y_pred = estimator.predict(X_test)
#     acc=accuracy_score(y_test, y_pred)
#     print(f"acc={acc}")
def score_estimator(estimator):
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    print(f"acc={acc},{estimator.__class__.__name__}")
##

# report=classification_report(y_test, y_pred)
# print(report)
##
