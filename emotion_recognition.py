##
# from typing_extensions import deprecated
import random
from time import time

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import tqdm

# from deprecated import deprecated
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GridSearchCV

from create_csv import write_emodb_csv, write_ravdess_csv
from data_extractor import load_data
from EF import (
    AHNPS,
    AVAILABLE_EMOTIONS,
    HNS,
    MCM,
    ava_emotions,
    e_config_def,
    f_config_def,
    get_f_config_dict,
    validate_emotions,
)
from MetaPath import (
    dbs,
    emodb,
    meta_dir,
    meta_paths_dbs,
    ravdess,
    test_emodb_csv,
    test_ravdess_csv,
    train_emodb_csv,
    train_ravdess_csv,
)
from utils import best_estimators, extract_feature


class EmotionRecognizer:
    """A class for training, testing and predicting emotions based on
    speech's features that are extracted and fed into `sklearn` or `keras` model"""

    def __init__(
        self,
        model=None,
        classification_task=True,
        dbs=None,
        e_config=None,
        f_config=None,
        train_meta_files=None,
        test_meta_files=None,
        balance=False,
        override_csv=True,
        verbose=1,
        **kwargs,
    ):
        """
        Params:
            model (sklearn model): the model used to detect emotions. If `model` is None, then self.determine_best_model()
                will be automatically called
                这个参数其实就是sklearn中的Estimator对象,例如SVC()示例化出来的对象
            emotions (list): list of emotions to be used. Note that these emotions must be available in
                RAVDESS & EMODB Datasets, available nine emotions are the following:
                    'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps' ( pleasant surprised ), 'boredom'.
                默认识别三种情感
                Default is ["sad", "neutral", "happy"].
            情感数据库的使用开关:
            ravdess (bool): whether to use RAVDESS Speech datasets, default is True
            emodb (bool): whether to use EMO-DB Speech dataset, default is True,
            custom_db (bool): whether to use custom Speech dataset that is located in `data/train-custom`
                and `data/test-custom`, default is True
            输出文件名的指定(应该指定为csv文件,即参数带有扩展后缀csv)
            ravdess_name (str): the name of the output CSV file for RAVDESS dataset, default is "ravdess.csv"
            emodb_name (str): the name of the output CSV file for EMO-DB dataset, default is "emodb.csv"
            custom_db_name (str): the name of the output CSV file for the custom dataset, default is "custom.csv"
            指定需要提取的情感特征,默认三种:mfcc,chroma,mel
            features (list): list of speech features to use, default is ["mfcc", "chroma", "mel"]
                (i.e MFCC, Chroma and MEL spectrogram )
            指定要使用的分类模型还是回归模型,默认使用分类模型
            classification (bool): whether to use classification or regression, default is True
            balance (bool): whether to balance the dataset ( both training and testing ), default is True
            verbose (bool/int): whether to print messages on certain tasks, default is 1
        Note that when `ravdess`, `emodb` and `custom_db` are set to `False`, `ravdess` will be set to True
        automatically.
        """
        # emotions
        self.e_config = e_config if e_config else e_config_def
        # make sure that there are only available emotions
        validate_emotions(self.e_config)
        self.f_config = f_config if f_config else f_config_def

        # 转换为字典格式(待优化)
        # @deprecated(version='1.0', reason='请使用 new_function() 代替')
        self._f_config_dict: dict[str, bool] = get_f_config_dict(self.f_config)
        self.train_meta_files = train_meta_files
        self.test_meta_files = test_meta_files
        # 可以使用python 默认参数来改造写法
        # 默认执行分类任务
        self.classification_task = classification_task
        self.balance = balance
        self.override_csv = override_csv
        self.verbose = verbose
        # boolean attributes
        self.balance = False
        self.data_loaded = False
        self.model_trained = False
        self.model = model if model is not None else self.best_model()

        self.dbs = dbs if dbs else [ravdess]
        # 鉴于数据集(特征和标签)在评估方法时将反复用到,因此这里将设置相应的属性来保存它们
        # 另一方面,如果模仿sklearn中的编写风格,其实是将数据和模型计算分布在不同的模块(类)中,比如
        # sklearn.datasets负责数据集生成
        # sklearn.model_selection负责划分数据集和训练集
        # sklearn.algorithms* 负责创建模型
        # sklearn.metrics 负责评估模型
        # 设置相应的属性的方便之处在于方法的调用可以少传参
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_audio_paths = None
        self.test_audio_paths = None

        # if self.model is None:
        # 依赖于boolean attributes

    # def prepare():

    def load_data(self):
        """
        导入指定的语料库数据,并提取特征
        Loads and extracts features from the audio files for the db's specified
        """
        # 判断是否已经导入过数据.如果已经导入,则跳过,否则执行导入
        if not self.data_loaded:
            # 调用data_extractor中的数据导入函数
            data = load_data(
                train_meta_files=self.train_meta_files,
                test_meta_files=self.test_meta_files,
                f_config=self.f_config,
                classification_task=self.classification_task,
                e_config=self.e_config,
                balance=self.balance,
            )
            # 设置实例的各个属性
            self.X_train = data["X_train"]
            self.X_test = data["X_test"]
            self.y_train = data["y_train"]
            self.y_test = data["y_test"]
            self.train_audio_paths = data["train_audio_paths"]
            self.test_audio_paths = data["test_audio_paths"]
            self.balance = data["balance"]
            if self.verbose:
                print("[+] Data loaded")
            self.data_loaded = True

    def train(self, verbose=1):
        """

        Train the model, if data isn't loaded, it 'll be loaded automatically

        X_train=None, y_train=None
        """
        if not self.data_loaded:
            # if data isn't loaded yet, load it then
            self.load_data()

        if not self.model_trained:
            X_train = self.X_train
            y_train = self.y_train
            self.model.fit(X=X_train, y=y_train)
            self.model_trained = True
            if verbose:
                print("[+] Model trained")

    def predict(self, audio_path):
        """
        预测单个音频的情感
        given an `audio_path`, this method extracts the features
        and predicts the emotion
        """
        feature = extract_feature(audio_path, **self._f_config_dict).reshape(1, -1)
        return self.model.predict(feature)[0]

    def predict_proba(self, audio_path):
        """
        Predicts the probability of each emotion.
        """
        if self.classification_task:
            feature = extract_feature(audio_path, **self._f_config_dict).reshape(1, -1)
            proba = self.model.predict_proba(feature)[0]
            result = {}
            for emotion, prob in zip(self.model.classes_, proba):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError(
                "Probability prediction doesn't make sense for regression"
            )

    def grid_search(self, params, n_jobs=2, verbose=3):
        """
        使用网格化搜索的方式搜索最优超参数
        Performs GridSearchCV on `params` passed on the `self.model`
        And returns the tuple: (best_estimator, best_params, best_score).
        """
        score = accuracy_score if self.classification_task else mean_absolute_error
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=params,
            scoring=make_scorer(score),
            n_jobs=n_jobs,
            verbose=verbose,
            cv=3,
        )
        # 调用fit开始传入数据集并搜索
        X_train, y_train = self.X_train, self.y_train
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train are None")
        # fit过程是一个耗时的过程
        grid_result = grid.fit(X_train, y_train)
        return (
            grid_result.best_estimator_,
            grid_result.best_params_,
            grid_result.best_score_,
        )

    def best_model(self):
        """
        从常见的模型中计算出最好的Estimator(model)
        Loads best estimators and determine which is best for test data,
        and then set it to `self.model`.
        # 使用MSE来评价回归模型,使用accuracy来评价分类模型
        In case of regression, the metric used is MSE(均方误差) and accuracy for classification.

        Note that the execution of this method may take several minutes due
        to training all estimators (stored in `grid` folder) for determining the best possible one.
        """
        if not self.data_loaded:
            self.load_data()

        # loads estimators
        estimators = best_estimators()

        result = []

        if self.verbose:
            # 控制是否显示进度条
            # 通过tqdm封装estimator这个可迭代对象,就可以在遍历estimator时,控制进度条的显示
            estimators = tqdm.tqdm(estimators)

        for estimator, params, cv_score in estimators:
            if self.verbose:
                estimators.set_description(f"Evaluating {estimator.__class__.__name__}")
            # self.model = estimator

            detector = EmotionRecognizer(
                model=estimator,
                emotions=self.e_config,
                classification_task=self.classification_task,
                features=self.f_config,
                balance=self.balance,
                override_csv=False,
            )
            # data already loaded
            detector.X_train = self.X_train
            detector.X_test = self.X_test
            detector.y_train = self.y_train
            detector.y_test = self.y_test
            detector.data_loaded = True
            # train(fit) the model
            # 如果设置verbose=1,则会逐个打印当前计算的模型(进度不是同一条)
            detector.train(verbose=0)

            # train(fit) the model
            # self.train(verbose=1)

            accuracy = detector.test_score()
            # print(f"[I] {estimator.__class__.__name__} with {accuracy} test accuracy")
            # append to result

            result.append((detector.model, accuracy))

        # sort the result
        # regression: best is the lower, not the higher
        # classification: best is higher, not the lower
        result = sorted(
            result, key=lambda item: item[1], reverse=self.classification_task
        )
        best_estimator = result[0][0]
        accuracy = result[0][1]

        self.model = best_estimator
        self.model_trained = True
        if self.verbose:
            if self.classification_task:
                print(
                    f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy*100:.3f}% test accuracy"
                )
            else:
                print(
                    f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy:.5f} mean absolute error"
                )
        return best_estimator

    def test_score(self):
        """
        Calculates score on testing data
        if `self.classification` is True, the metric used is accuracy,
        Mean-Squared-Error is used otherwise (regression)
        """
        X_test = self.X_test
        y_test = self.y_test
        y_pred = self.model.predict(X_test)
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test are None")

        if self.classification_task:
            res = accuracy_score(y_true=y_test, y_pred=y_pred)
        else:
            res = mean_squared_error(y_true=y_test, y_pred=y_pred)
        if self.verbose >= 2:
            report = classification_report(y_true=y_test, y_pred=y_pred)
            print(report, self.model.__class__.__name__)
        return res

    def train_score(self, X_train, y_train):
        """
        Calculates accuracy score on training data
        if `self.classification` is True, the metric used is accuracy,
        Mean-Squared-Error is used otherwise (regression)
        """
        y_pred = self.model.predict(X_train, y_train)
        if self.classification_task:
            return accuracy_score(y_true=y_train, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=y_train, y_pred=y_pred)

    def train_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_train)
        y_train = self.y_train
        if y_train is None:
            raise ValueError("y_train is None")

        return fbeta_score(y_true=y_train, y_pred=y_pred, beta=beta, average="micro")

    def test_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_test)
        y_test = self.y_test
        if y_test is None:
            raise ValueError("y_test is None")
        return fbeta_score(y_true=y_test, y_pred=y_pred, beta=beta, average="micro")

    def confusion_matrix(self, percentage=True, labeled=True):
        """
        Computes confusion matrix to evaluate the test accuracy of the classification
        and returns it as numpy matrix or pandas dataframe (depends on params).
        params:
            percentage (bool): whether to use percentage instead of number of samples, default is True.
            labeled (bool): whether to label the columns and indexes in the dataframe.
        """
        if not self.classification_task:
            raise NotImplementedError(
                "Confusion matrix works only when it is a classification problem"
            )
        y_pred = self.model.predict(self.X_test)
        y_test = self.y_test
        if y_test is None:
            raise ValueError("y_test is None")
        matrix = confusion_matrix(
            y_true=y_test, y_pred=y_pred, labels=self.e_config
        ).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(
                matrix,
                index=[f"true_{e}" for e in self.e_config],
                columns=[f"predicted_{e}" for e in self.e_config],
            )
        return matrix

    def draw_confusion_matrix(self):
        """Calculates the confusion matrix and shows it"""
        matrix = self.confusion_matrix(percentage=False, labeled=False)
        # TODO: add labels, title, legends, etc.
        pl.imshow(matrix, cmap="binary")
        pl.show()

    def get_n_samples(self, emotion, partition):
        """Returns number data samples of the `emotion` class in a particular `partition`
        ('test' or 'train')
        """
        res = 0
        if partition == "test":
            if self.y_test is None:
                raise ValueError("y_test is None")
            res = len([y for y in self.y_test if y == emotion])
        elif partition == "train":
            if self.y_train is None:
                raise ValueError("y_train is None")
            res = len([y for y in self.y_train if y == emotion])
        return res

    def get_samples_by_class(self):
        """
        Returns a dataframe that contains the number of training
        and testing samples for all emotions.
        Note that if data isn't loaded yet, it'll be loaded
        """
        if not self.data_loaded:
            self.load_data()
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.e_config:
            n_train = self.get_n_samples(emotion, "train")
            n_test = self.get_n_samples(emotion, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)

        # get total
        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(
            data={"train": train_samples, "test": test_samples, "total": total},
            index=self.e_config + ["total"],
        )

    def get_random_emotion(self, emotion, partition="train"):
        """
        Returns random `emotion` data sample index on `partition`.
        """
        if partition == "train":
            index = random.choice(list(range(len(self.y_train))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_train))))
        elif partition == "test":
            index = random.choice(list(range(len(self.y_test))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_test))))
        else:
            raise TypeError("Unknown partition, only 'train' or 'test' is accepted")

        return index


def plot_histograms(classifiers=True, beta=0.5, n_classes=3, verbose=1):
    """
    Loads different estimators from `grid` folder and calculate some statistics to plot histograms.
    Params:
        classifiers (bool): if `True`, this will plot classifiers, regressors otherwise.
        beta (float): beta value for calculating fbeta score for various estimators.
        n_classes (int): number of classes
    """
    # get the estimators from the performed grid search result
    estimators = best_estimators(classifiers)

    final_result = {}
    for estimator, params, cv_score in estimators:
        final_result[estimator.__class__.__name__] = []
        for i in range(3):
            result = {}
            # initialize the class
            detector = EmotionRecognizer(estimator, verbose=0)
            # load the data
            detector.load_data()
            if i == 0:
                # first get 1% of sample data
                sample_size = 0.01
            elif i == 1:
                # second get 10% of sample data
                sample_size = 0.1
            elif i == 2:
                # last get all the data
                sample_size = 1
            # calculate number of training and testing samples
            n_train_samples = int(len(detector.X_train) * sample_size)
            n_test_samples = int(len(detector.X_test) * sample_size)
            # set the data
            detector.X_train = detector.X_train[:n_train_samples]
            detector.X_test = detector.X_test[:n_test_samples]
            detector.y_train = detector.y_train[:n_train_samples]
            detector.y_test = detector.y_test[:n_test_samples]
            # calculate train time
            t_train = time()
            detector.train()
            t_train = time() - t_train
            # calculate test time
            t_test = time()
            test_accuracy = detector.test_score()
            t_test = time() - t_test
            # set the result to the dictionary
            result["train_time"] = t_train
            result["pred_time"] = t_test
            result["acc_train"] = cv_score
            result["acc_test"] = test_accuracy
            result["f_train"] = detector.train_fbeta_score(beta)
            result["f_test"] = detector.test_fbeta_score(beta)
            if verbose:
                print(
                    f"[+] {estimator.__class__.__name__} with {sample_size*100}% ({n_train_samples}) data samples achieved {cv_score*100:.3f}% Validation Score in {t_train:.3f}s & {test_accuracy*100:.3f}% Test Score in {t_test:.3f}s"
                )
            # append the dictionary to the list of results
            final_result[estimator.__class__.__name__].append(result)
        if verbose:
            print()
    visualize(final_result, n_classes=n_classes)


def visualize(results, n_classes):
    """
    Visualization code to display results of various learners.

    inputs:
      - results: a dictionary of lists of dictionaries that contain various results on the corresponding estimator
      - n_classes: number of classes
    """

    n_estimators = len(results)

    # naive predictor
    accuracy = 1 / n_classes
    f1 = 1 / n_classes
    # Create figure
    fig, ax = pl.subplots(2, 4, figsize=(11, 7))
    # Constants
    bar_width = 0.4
    colors = [
        (random.random(), random.random(), random.random()) for _ in range(n_estimators)
    ]
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(
            ["train_time", "acc_train", "f_train", "pred_time", "acc_test", "f_test"]
        ):
            for i in np.arange(3):
                x = bar_width * n_estimators
                # Creative plot code
                ax[j // 3, j % 3].bar(
                    i * x + k * (bar_width),
                    results[learner][i][metric],
                    width=bar_width,
                    color=colors[k],
                )
                ax[j // 3, j % 3].set_xticks([x - 0.2, x * 2 - 0.2, x * 3 - 0.2])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.2, x * 3))
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(
        y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[1, 1].axhline(
        y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[0, 2].axhline(
        y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[1, 2].axhline(
        y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    # Set additional plots invisibles
    ax[0, 3].set_visible(False)
    ax[1, 3].axis("off")
    # Create legend
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()
    # Aesthetics
    pl.suptitle(
        "Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10
    )
    pl.tight_layout()
    pl.show()


from MetaPath import pair1, select_meta_dict, test_emodb_csv, train_emodb_csv

if __name__ == "__main__":
    # from emotion_recognition import EmotionRecognizer
    from sklearn.svm import SVC

    # use SVC as a demo
    my_model = SVC()
    # pass model to EmotionRecognizer instance
    # and balance the dataset
    # train the model
    # train_meta_files= meta_paths_dbs(partition="train",dbs=db)
    # test_meta_files =meta_paths_dbs(partition="test",dbs=db)

    train_meta_files = train_ravdess_csv
    test_meta_files = test_ravdess_csv

    # rec = EmotionRecognizer(train_meta_files=train_meta_files,test_meta_files=test_meta_files,model=my_model, verbose=1)

    from MetaPath import pair2, pair3, pair4

    meta_files_dict = select_meta_dict(pair3)
    rec = EmotionRecognizer(**meta_files_dict, verbose=1)

    # data=rec.load_data()

    # data = load_data(
    #     train_meta_files=train_meta_files,
    #     test_meta_files=test_meta_files,
    #     balance=False,
    # )

    # X_train = data["X_train"]
    # y_train = data["y_train"]
    # rec.train(X_train=X_train, y_train=y_train)
