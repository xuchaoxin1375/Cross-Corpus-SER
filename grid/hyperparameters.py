from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


(SVC(C=10, gamma=0.001), {"C": 10, "gamma": 0.001, "kernel": "rbf"}, 0.9381835473133618)
(
    RandomForestClassifier(max_depth=7, max_features=0.5, n_estimators=40),
    {
        "max_depth": 7,
        "max_features": 0.5,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 40,
    },
    0.8854018069424631,
)
(
    GradientBoostingClassifier(learning_rate=0.3, max_depth=7, subsample=0.7),
    {
        "learning_rate": 0.3,
        "max_depth": 7,
        "max_features": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 100,
        "subsample": 0.7,
    },
    0.9476937708036139,
)
(
    KNeighborsClassifier(n_neighbors=3, p=1, weights="distance"),
    {"n_neighbors": 3, "p": 1, "weights": "distance"},
    0.9320019020446981,
)
(
    MLPClassifier(
        alpha=0.01,
        batch_size=512,
        hidden_layer_sizes=(300,),
        learning_rate="adaptive",
        max_iter=400,
    ),
    {
        "alpha": 0.01,
        "batch_size": 512,
        "hidden_layer_sizes": (300,),
        "learning_rate": "adaptive",
        "max_iter": 400,
    },
    0.9358059914407989,
)
(
    BaggingClassifier(max_features=0.5, n_estimators=50),
    {"max_features": 0.5, "max_samples": 1.0, "n_estimators": 50},
    0.9210651450309082,
)
(
    DecisionTreeClassifier(criterion="entropy", max_depth=7, max_features="sqrt"),
    {
        "criterion": "entropy",
        "max_depth": 7,
        "max_features": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
    },
    "overfit",
)
(
    AdaBoostClassifier(algorithm="SAMME", learning_rate=0.8, n_estimators=60),
    {"algorithm": "SAMME", "learning_rate": 0.8, "n_estimators": 60},
    "",
)
(
    StackingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(max_depth=3, n_estimators=10, random_state=1),
            ),
            (
                "adab",
                AdaBoostClassifier(learning_rate=0.1, n_estimators=10, random_state=1),
            ),
        ],
        final_estimator=LogisticRegression(),
    ),
    "",
    "",
)
["bclf.joblib"]
