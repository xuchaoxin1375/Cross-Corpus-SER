
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



# 加载iris数据集
iris = load_iris()

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# 定义KNN模型并进行训练
knn = KNeighborsClassifier(n_neighbors=3,p=1, weights='distance') # K值为3
knn.fit(X_train, y_train)

# 对测试集进行预测
y_pred = knn.predict(X_test)

# 输出预测结果
print("预测结果：", y_pred)

cm=confusion_matrix(y_test,y_pred=y_pred)
print(cm,"@{cm}")
accu=accuracy_score(y_true=y_test,y_pred=y_pred)
print(accu,"@{accu}")
report=classification_report(y_true=y_test,y_pred=y_pred)
print(report,"@{report}")