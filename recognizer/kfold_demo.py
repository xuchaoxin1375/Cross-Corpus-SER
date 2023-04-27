##
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier

##

# 加载iris(鸢尾花)数据集
X, y = load_iris(return_X_y=True)

#! 定义5折交叉验证
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# 使用线性回归模型进行训练和测试
model = LinearRegression()
# model=RandomForestClassifier()

scores_cv = []
# 这里split参数可以是X也可以是y,因为只需要划分样本的索引,所以两者都可以
for train_index, test_index in kf.split(y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores_cv.append(score)
    print("Score:", score)
mean_score = np.mean(scores_cv)
print(f"{mean_score=}")
##
#!使用cross_val_score
#构造cv器的时候不需要传入数据集
ss_cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
kf_cv=KFold(n_splits=3,shuffle=True,random_state=42)
scores = cross_val_score(
    model,
    X,
    y,
    #cv=5,
    #cv=ss_cv,
    cv=kf_cv,
    verbose=3,
)
#cv取整数时,采用的非随机化的kfold方法划分,不是很可靠
#cv建议选用随机化的(StratifiedShuffleSplit最为高级)
#cv取kfold对象时,我们可以选择shuffle=True,使得所有样本都能够参与训练集/测试集
print("Scores:", scores)
print("Mean score:", scores.mean())


##

# 使用决策树模型进行交叉验证，并对数据集进行随机化操作
model = DecisionTreeClassifier()
ss_cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

print("cv: ", ss_cv)

# ssr=ss_cv.split(X,y)
# for train_index,test_index in ssr:
#     train_index,test_index=np.array(train_index),np.array(test_index)
#     print(train_index.shape,test_index.shape)

scores = cross_val_score(model, X, y, cv=ss_cv, verbose=True)
print("Scores:", scores)
print("Mean score:", scores.mean())


##

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# 使用随机生成数据测试
rng = np.random.default_rng()
rng.integers(20, size=(12, 2))
# 样本总数为12,二分类,标签为0,1,两种样本比例为1:2
n = 12
n0, n1 = 1 * n // 3, 2 * n // 3
# 随机的为这些模拟样本分配标签(因为这里不涉及到训练,所以随机分配标签不影响效果,在数据集划分的阶段,不用关心样本和标签的关联规律,如果是要训练,通常是不能随机给样本特征分配标签)

y = [0] * n0 + [1] * n1
y = np.array(y)
rng.shuffle(y)
# 下面一种方式采用概率的方式生成标签,但是即使样本总数n可以被3整除,生成的数组也不保证数量是1:2
# y = rng.choice([0, 1], size=12, replace=True, p=[1/3, 2/3])
# count=np.unique(y,return_counts=True)
# print(count)
# 为例放便验证,这里将标签数组和样本索引打印出来
print(np.vstack([y, range(n)]))
# 构造分层随机拆分对象,指定做独立的5次划分,每次划分,测试集的样本数量占总样本数量n的20%
# 而StratifiedShuffleSplit会保持各个类别在测试集和训练集上的比例
# 是两种独立的约束(例如0类样和1类样本比例在数据集中为1:2,那么在训练集和测试集中依然保持(或接近)1:2)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# print(sss)
# 打印这5次
for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    print(np.vstack((test_index, y[test_index])))
##
X,y=load_iris(return_X_y=True)
#效果等同于
data=load_iris()
X=data.data
y=data.target
#由于Bunch对象的特性,可以用字典方式访问
X=data['data']
y=data['target']
# 
# 导出为pandas dataframe:
frame_data=load_iris(as_frame=True)
X_df=frame_data.data
y_df=frame_data.target
