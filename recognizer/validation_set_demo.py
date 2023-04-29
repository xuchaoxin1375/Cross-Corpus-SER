from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载iris数据集
iris = load_iris()

# 提取特征和标签
X = iris.data
y = iris.target

# 将数据集分割为训练集、验证集和测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将训练集和验证集的组合再次分割为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

# 打印分割后的数据集大小
print('训练集大小:', X_train.shape[0])
print('验证集大小:', X_val.shape[0])
print('测试集大小:', X_test.shape[0])