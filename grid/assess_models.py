from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from recognizer.basic import EmotionRecognizer

# 定义决策树参数空间
param_grid = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None, 5, 7, 10, 13],
    "min_samples_split": [1,2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
AdaBoostClassifier
# 初始化决策树分类器
dt = DecisionTreeClassifier()
er=EmotionRecognizer(

)
# 使用GridSearchCV搜索最优参数
grid_search = GridSearchCV(dt, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 使用RandomizedSearchCV搜索最优参数
random_search = RandomizedSearchCV(dt, param_distributions=param_grid, cv=5, n_iter=10)
random_search.fit(X_train, y_train)
