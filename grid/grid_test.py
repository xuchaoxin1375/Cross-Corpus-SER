##
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,BaggingRegressor
from sklearn.model_selection import RandomizedSearchCV
from joblib import load, dump
best_clf = "best_clf.joblib"
best_rgr = "best_rgr.joblib"
# bclf_job=dump(bclf,'best_clf.joblib')
# brgr_job=dump(brgr,'best_rgr.joblib')

bclf = load(best_clf)
brgr = load(best_rgr)
##
b = BaggingClassifier(max_features=0.5, n_estimators=50)
print(b)
bag_best = (BaggingClassifier(max_features=0.5, n_estimators=50),
            {'max_features': 0.5, 'max_samples': 1.0, 'n_estimators': 50},
            0.9210651450309082)
bclf[-1] = bag_best

# tuple(x[0])
##
for item in bclf:
    print(item)

##
bag_rgr_best = (BaggingRegressor(max_features=1, max_samples=0.1),
                {'max_features': 1, 'max_samples': 0.1, 'n_estimators': 10},
                0.6521001743540973)
brgr[-1]=bag_rgr_best
##
# 生成二分类数据集
X, y = make_classification(n_samples=500, n_features=5,
                           n_informative=3, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)
# 查看数据集的形状和标签分布
print(X.shape)
print(y[:10])
##


# create a pipeline with scaling and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVR())
])

# define the parameter space for grid search
param_grid = {
    # 'svm__kernel':['linear','rbf'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.01, 0.1, 1],
}

# create a grid search object and fit to the data
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

# evaluate the best model on the test set
grid_search.score(X_test, y_test)
##

X, y = fetch_california_housing(return_X_y=True)
iris = load_iris()
X, y = iris.data, iris.target  # type: ignore
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

##
# define the parameter space that will be searched over
param_distributions = {'n_estimators': randint(1, 5),
                       'max_depth': randint(3, 6)}

# now create a searchCV object and fit it to the data
# 根据超参数空间,构造CV对象
RFsearch = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                              n_iter=5,
                              param_distributions=param_distributions,
                              verbose=3,
                              random_state=0)
# 开始搜索(调用SearchCV对象的fit方法)
RFsearch.fit(X_train, y_train)
# 从搜索结果中获取最优参数
print(RFsearch.best_params_, "@{search.best_params_}")
# 计算得分
RFsearch.score(X_test, y_test)

##
param_grid = {
    'n_estimators': [1, 2, 3],
    'max_depth': [3, 4, 5, 6]
}

# Create a GridSearchCV object and fit it to the data
Gsearch = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                       param_grid=param_grid,
                       cv=3,
                       verbose=3)
Gsearch.fit(X_train, y_train)

print(f"f{Gsearch.best_params_}")
Gsearch.score(X_test, y_test)
##
