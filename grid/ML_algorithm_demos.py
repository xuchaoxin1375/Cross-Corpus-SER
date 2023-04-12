##
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

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

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 输出分类器的准确率
print("Accuracy:", clf.score(X_test, y_test))
## 
#直接使用交叉验证

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
res=cross_validate(clf,X_train, y_train)
# print(res)
for item in res.items():
    print(item)
##
