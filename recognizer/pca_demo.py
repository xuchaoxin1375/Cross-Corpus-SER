##
from sklearn.datasets import fetch_openml,load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载数据集
# mnist = load('mnist_784', version=1)
mnist=load_digits()
X = mnist.data
y = mnist.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 不使用PCA进行训练和测试
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print("Accuracy without PCA: {:.2f}%".format(accuracy * 100))

# 使用PCA进行训练和测试
pca = PCA(n_components=None)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
knn_pca = KNeighborsClassifier()
knn_pca.fit(X_train_pca, y_train)
accuracy_pca = knn_pca.score(X_test_pca, y_test)
print("Accuracy with PCA: {:.2f}%".format(accuracy_pca * 100))


#对比降维后的维数
print(X_test_pca.shape,"@shape of {X_test_pca}")
print(X_test.shape,"@shape of {X_test}")


##
import numpy as np
from sklearn.decomposition import PCA

rng=np.random.default_rng()
X=rng.integers(10,size=(15,10))


def check_attributes_of_pca(n_components='mle',svd_solver='auto'):
    pca = PCA(n_components=n_components,svd_solver=svd_solver)
    # 训练PCA模型，并对样本进行降维
    X_pca = pca.fit_transform(X)
    # 查看PCA模型的各个属性
    print("PCA模型的主成分数：", pca.n_components_)
    print("PCA模型的主成分：", pca.components_)
    print("PCA模型的各主成分的方差值：", pca.explained_variance_)
    print("PCA模型各主成分方差值所占比例：", pca.explained_variance_ratio_)
    print("PCA模型的均值：", pca.mean_)
    print("PCA模型的噪声方差：", pca.noise_variance_)
    print("降维后的样本矩阵：\n", X_pca)

for nc in ['mle',2,5,None]:
    check_attributes_of_pca(n_components=nc)
    print("-"*20)
##

