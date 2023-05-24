from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split as tts
import numpy as np

X, y = load_diabetes(return_X_y=True)

X = X[:, np.newaxis, 2]
# 利用np.newaxis,2 将10维特征向量的第2维打包一层,相当于插入了一个轴,该轴上的维数是1
# X.shape(442,1)
X_train, X_test, y_train, y_test = tts(X, y, test_size=20, shuffle=False)

from sklearn.linear_model import Ridge, LinearRegression

# 控制使用高清格式(svg)
from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats("svg")


def linear_score_plot(model, **kwargs):
    """基于plot风格编写的函数,如果多次调用,图像将叠加在通过一个坐标系中

    Parameters
    ----------
    model : sklearn.estimator
        估计器

    Returns
    -------
    matplotlib.pyplot
        添加了图像的plt
        不过,由于这里使用的是plot风格,因此可以直接使用import matplotlib.pyplot as plt处的plt,效果是一样的
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("train_score: ", train_score)
    print("test_score: ", test_score)

    print(kwargs)
    plt.plot(X_test, y_pred, **kwargs)
    plt.legend()
    return plt


# 添加一条岭回归预测线
rd = Ridge()
linear_score_plot(rd, label="Ridge")
# plt1.show()
## 添加逻辑回归的预测直线
lr = LinearRegression()
linear_score_plot(lr, label="LinearRegression")
# 绘制一层数据集中对应散点图

plt.scatter(X_test, y_test, label="dataset points")
#将label作为图例显示出来
plt.legend()
# 将图像显示出来
plt.show()
