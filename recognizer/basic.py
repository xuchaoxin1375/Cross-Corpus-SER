##
# from typing_extensions import deprecated
import warnings
from sklearn.tree import DecisionTreeClassifier
import config.MetaPath as mp
from config.MetaPath import test_emodb_csv
from config.algoparams import ava_ML_algorithms
import random
from time import time
from config.algoparams import ava_cv_modes
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from audio.extractor import AudioExtractor, load_data_from_meta
import config.EF as ef
from config.EF import AHNPS, e_config_def, f_config_def, validate_emotions
from config.MetaPath import (
    emodb,
    meta_paths_of_db,
    ravdess,
    savee,
    validate_partition,
    project_dir,
)
import config.MetaPath as meta
from audio.core import best_estimators, extract_feature_of_audio
from config.algoparams import random_state


##
class EmotionRecognizer:
    """A class for training, testing ,predicting,anaylzing emotions based on
    speech's features that are extracted and fed into `sklearn` model

    examples
    -
    rec = EmotionRecognizer(model=my_model,e_config=AHNPS,f_config=f_config_def,test_dbs=[ravdess],train_dbs=[ravdess], verbose=1)

    rec = EmotionRecognizer(model=my_model,e_config=AHNPS,f_config=f_config_def,test_dbs=emodb,train_dbs=emodb, verbose=1)
    """

    def __init__(
        self,
        model=None,
        classification_task=True,
        dbs=None,
        e_config=None,
        f_config=None,
        train_dbs=None,
        test_dbs=None,
        balance=False,
        shuffle=True,
        override_csv=True,
        cross=False,  # 表示跨库(这回让其读取train_db_econfig.csv作为测试集,样例更丰富,更合理,当然还可以进一步改进,读取all_db_econfg.csv,但是这里暂不执行)
        verbose=0,
        **feature_transforms,
        # **kwargs,
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
        # self._f_config_dict: dict[str, bool] = get_f_config_dict(self.f_config)
        self.train_dbs = train_dbs
        self.test_dbs = test_dbs

        # print(self.train_meta_files, self.test_meta_files)
        self.feature_transforms = feature_transforms

        # 可以使用python 默认参数来改造写法
        # 默认执行分类任务
        self.classification_task = classification_task
        self.verbose = verbose
        # boolean attributes
        self.override_csv = override_csv
        self.shuffle = shuffle
        self.balance = balance
        self.cross = cross
        # 非构造器初始化变量
        self.data_loaded = False
        self.model_trained = False
        self.ae = None
        self.dbs = dbs if dbs else [ravdess]
        # 鉴于数据集(特征和标签)在评估方法时将反复用到,因此这里将设置相应的属性来保存它们
        # 另一方面,如果模仿sklearn中的编写风格,其实是将数据和模型计算分布在不同的模块(类)中,比如
        # sklearn.datasets负责数据集生成
        # sklearn.model_selection负责划分数据集和训练集
        # sklearn.algorithms* 负责创建模型
        # sklearn.metrics 负责评估模型
        # 设置相应的属性的方便之处在于方法的调用可以少传参
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.y_pred = []
        self.train_audio_paths = []
        self.test_audio_paths = []
        # 开始填充数据(最先开始的步骤,放在init中随着初始化实例的时候执行)
        # self.load_data()
        # 属性的先后位置会影响程序的运行
        # print("@{model}")
        # print(model,"\ncomparing and choosing the best model...")
        # !RandomForestClassifier实例不能直接用bool()来判断,会提示estimators_不存在
        self.model = model
        # if self.model is None:
        # 依赖于boolean attributes

        self.train_meta_files = meta_paths_of_db(
            db=self.train_dbs,
            e_config=self.e_config,
            change_type="str",
            partition="train",
        )
        # 判断跨库任务
        test_meta_partition = "test"
        if self.cross:
            test_meta_partition = "train"
        self.test_meta_files = meta_paths_of_db(
            db=self.test_dbs,
            e_config=self.e_config,
            change_type="str",
            partition=test_meta_partition,
        )

    # def prepare():

    def load_data(self):
        """
        导入指定的语料库数据,并提取特征
        Loads and extracts features from the audio files for the db's specified
        - 注意,由于balance操作对于数据集划分有一定要求,不是任何数据集都可以执行balance操作(例如三分类中,test set中只缺失了某一个类别的样本,这中情况下执行balance,将导致测试集样本数量为空)
        """
        # 判断是否已经导入过数据.如果已经导入,则跳过,否则执行导入
        if not self.data_loaded:
            # 调用extractor中的数据导入函数
            data = load_data_from_meta(
                train_meta_files=self.train_meta_files,
                test_meta_files=self.test_meta_files,
                f_config=self.f_config,
                e_config=self.e_config,
                classification_task=self.classification_task,
                balance=self.balance,
                shuffle=self.shuffle,
                feature_transforms=self.feature_transforms,
                verbose=self.verbose,
            )
            # 设置实例的各个属性
            # 事实上,也可以直接用load_data_from_meta返回的结果中的ae对象,赋值ER对象(self.ae=data["ae"])
            self.ae = data["ae"]
            self.X_train = data["X_train"]
            self.X_test = data["X_test"]
            self.y_train = data["y_train"]
            self.y_test = data["y_test"]
            self.train_audio_paths = data["train_audio_paths"]
            self.test_audio_paths = data["test_audio_paths"]

            self.balanced_success(data)
            if self.verbose:
                print("[I] Data loaded\n")
                print(f"{self.ae=}")
                print(f"{self.ae.pca=}🎈")
            self.data_loaded = True
            # print(id(self))
            if self.verbose > 1:
                print(vars(self))

    def balanced_success(self, res):
        self.balance = res["balance"]

    def train(self, choosing=False, verbose=1):
        """
        载入数据并训练模型(sklearn.estimator.fit)
        Train the model, if data isn't loaded, it will be loaded automatically

        X_train=None, y_train=None
        """
        if not self.data_loaded:
            # if data isn't loaded yet, load it then
            self.load_data()
        if self.verbose > 1:
            print("@{self.model}:")
            print(self.model)
        model = self.model if self.model is not None else self.best_model()
        if not self.model_trained or choosing:
            X_train = self.X_train
            y_train = self.y_train
            # estimator训练(fit)模型
            model.fit(X=X_train, y=y_train)
            self.model_trained = True
        if self.verbose > 1:
            if choosing == True:
                print(
                    f"[I] Model trained with{choosing=},choosing the best model,override the trained model.."
                )

    def predict(self, audio_path):
        """
        预测单个音频的情感
        由于是单个音频的情感预测,因此不需要考虑shuffle和balance这些操作,只需要提取语音特征,然后进行调用模型预测即可
        given an `audio_path`, this method extracts the features
        and predicts the emotion

        以下语句不再适合具有pca降维操作下的情形
        feature_audio = extract_feature_of_audio(audio_path, self.f_config)
        print(feature1.shape)
        print(feature1,"@{feature1}",feature1.shape)
        feature2=feature1.T
        print(feature2,"@{feature2}",feature2.shape)
        print(feature3,"@{feature3}",feature3.shape)
        """
        feature_audio = self.extract_feature_single_audio(audio_path)

        feature = feature_audio.reshape(1, -1)
        model = self.model if self.model else self.best_model()
        res = model.predict(feature)
        # res可能是个列表
        # print(res, "@{res}")
        return res[0]

    def extract_feature_single_audio(self, audio_path):
        """extract a single audio file feature

        Parameters
        ----------
        audio_path : path
            audio path

        Returns
        -------
        ndarray
            audio feature
        """
        ae: AudioExtractor = self.ae
        pca = ae.pca
        print(pca, "@{pca} in 'predict' method")
        # if pca:
        #     feature_audio=pca.transform(feature_audio)
        #     print(feature_audio.shape, "@{feature_audio.shape}")
        feature_audio = ae.extract_features(partition="test", audio_paths=[audio_path])
        return feature_audio
        # return self.model.predict(feature2)[0]

    def peek_test_set(self, n=5):
        res = [
            self.test_audio_paths[:n],
            self.X_test[:n],
            self.y_test[:n],
            self.y_pred[:n],
        ]
        return res

    def predict_proba(self, audio_path):
        """
        Predicts the probability of each emotion.
        """
        if self.classification_task:
            # feature = extract_feature_of_audio(audio_path, self.f_config).reshape(1, -1)
            feature = self.extract_feature_single_audio(audio_path)
            proba = self.model.predict_proba(feature)[0]
            result = {}
            for emotion, prob in zip(self.model.classes_, proba):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError(
                "Probability prediction doesn't make sense for regression"
            )

    def show_second(self):
        peeker = er.peek_test_set(2)
        feature = peeker[1][1]
        audio_path = peeker[0][1]
        feature_pred = self.predict(audio_path)
        print(feature[:5], feature_pred[:5])

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
        计算最优model时,也可以考虑创建新的ER实例来做计算最优model的用途,但会增加开销

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
            estimators = tqdm(estimators)

        for epc in estimators:
            estimator, params_, cv_score_ = epc
            ecn = estimator.__class__.__name__

            if self.verbose:
                # 如果启用verbose选项,那么estimators会被tqdm包装
                # 此时可以通过set_description方法来修改进度条的描述信息
                # 比如,estimators.set_description(f"Evaluating {estimator.__class__.__name__}")
                estimators.set_description(f"Evaluating <{ecn}>")
            # 为例避免相互干扰,每测试模型就创建一个ER对象(er)
            er = EmotionRecognizer(
                model=estimator,
                emotions=self.e_config,
                classification_task=self.classification_task,
                f_config=self.f_config,
                balance=self.balance,
                override_csv=False,
                verbose=0,
            )
            # data already loaded
            er.X_train = self.X_train
            er.X_test = self.X_test
            er.y_train = self.y_train
            er.y_test = self.y_test
            er.data_loaded = True
            # train the model
            er.train(verbose=0)
            # get test accuracy
            accuracy = er.test_score()
            # append to result
            result.append((er.model, accuracy))

            # 方法2:(小心使用)
            # 使用本对象self而不是在创建一个ER对象
            # self.model = estimator
            # er = self
            # 以下的计算是用来选出model的,而不是直接作为self对象的属性,这里将self赋值给er,以示区别
            # train(fit) the model
            # 如果设置verbose=1,则会逐个打印当前计算的模型(进度不是同一条)
            # er.train(choosing=True, verbose=0)
            # train(fit) the model
            # self.train(verbose=1)
            # accuracy = er.test_score(choosing=True)
            # append to result
            # result.append((estimator, accuracy))

            print(f"\n[I] {ecn} with {accuracy} test accuracy")

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
                    f"[🎈] Best model : {self.model.__class__.__name__} with {accuracy * 100:.3f}% test accuracy"
                )
            else:
                print(
                    f"[I] Best model : {self.model.__class__.__name__} with {accuracy:.5f} mean absolute error"
                )
        return best_estimator

    def test_score(self, choosing=False, verbose=0, report=False):
        """
        Calculates score on testing data
        Please call the `train` method before call this method.

        just like sklearn convention:call estimator.call `fit` at first,then call `predict` or `score` method

        if `self.classification` is True, the metric used is accuracy,
        Mean-Squared-Error is used otherwise (regression)

        1.调用训练好的模型进行预测

        2.如果model是None,那么调用best_model获取最优模型(这个过程会遍历一个可用模型列表,是通过调用ER实例的test_score()方法来计算,不过这里不会遇到None的情况,因此间接递归调用不超过2层)

        3.如果是best_model方法返回的model作为最终的model,那么本方法会对选出的模型再次训练并预测(对于带有随机性的算法,best_model的分数可能和本方法重新计算的分数不一致,但通常不会差太多(10%以内))
        """
        X_test = self.X_test
        y_test = self.y_test

        model = self.model if self.model is not None else self.best_model()
        self.validate_empty_array(X_test=X_test, y_test=y_test)

        # 预测计算
        if verbose > 1:
            print(X_test.shape, y_test.shape, "🎈")
        # 根据当前模型进行预测(直接调用estimator的predict方法)
        y_pred = model.predict(X_test)  # type: ignore
        # 如果处于best_model的过程中调用本方法(choosing=True),则上述预测内容作为临时结果不写入对象属性保存;否则作为最终结果写入属性保存
        # 默认choosing=False,也就是将结果保存到对象属性中
        if choosing == False:
            self.y_pred = np.array(y_pred)

        if self.classification_task:
            res = accuracy_score(y_true=y_test, y_pred=y_pred)
            # 结果和:er.model.score(er.X_test,er.y_test)一样,但是这种做法回独立将X_test预测一遍,而不保存预测结果,只给出得分
        else:
            res = mean_squared_error(y_true=y_test, y_pred=y_pred)
        if report:
            self.check_report(y_test, y_pred)
        return res

    def check_report(self):
        """输出模型当前结果的多个指标报告
        对于分类任务,包括precision(查准率),recall(回召或查全率),f1-score以及各类别的样本数量
        对于多分类,还有一些综合的指标(macro,weighted),每个单元格结合两个维度的表头进行理解和阅读

        由于跨库实验比较困难,有的样本类别无法被正确分类(所有该类别都被错误分类),此时classification_report方法会提出警告,除非使用zero_division参数替换掉默认的warn.
        """
        y_test = self.y_test
        y_pred = self.y_pred

        report = classification_report(y_true=y_test, y_pred=y_pred, zero_division=0)
        # print(report, self.model.__class__.__name__)
        return report

    def model_cv_score(
        self,
        choosing=False,
        verbose=1,
        mean_only=True,
        n_splits=5,
        test_size=0.2,
        cv_mode="sss",
    ):
        """
        使用交叉验证的方式来评估模型
        Calculates score on testing data
        """
        X_train = self.X_train
        y_train = self.y_train
        # 调用训练好的模型进行预测
        model = self.model if self.model is not None else self.best_model()
        self.validate_empty_array(X_train, y_train)

        # 预测计算
        if verbose > 1:
            print(X_train.shape, y_train.shape, "🎈")
            print(f"{n_splits=}")
        n_splits = int(n_splits)

        y_pred = model.predict(X_train)  # type: ignore
        if choosing == False:
            self.y_pred = np.array(y_pred)
        # 交叉验证的方式评估模型的得分
        cv_mode_dict = dict(
            sss=StratifiedShuffleSplit(
                n_splits=n_splits, test_size=test_size, random_state=0
            ),
            ss=ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0),
            kfold=KFold(n_splits=n_splits, shuffle=True, random_state=0),
            skfold=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0),
        )
        cv_mode_selected = cv_mode_dict[cv_mode]
        if verbose > 1:
            print(f"{cv_mode=}🎈")
        res = [0]
        if self.classification_task:
            # res = accuracy_score(y_true=y_test, y_pred=y_pred)
            # 将交叉验证器cv传递给cross_val_score函数执行评估操作
            # 而非自己使用来完成k折交叉验证
            # 所有对象都有__class__属性以及__name__二级属性
            if model.__class__.__name__ in ava_ML_algorithms:
                res = cross_val_score(model, X_train, y_train, cv=cv_mode_selected)
                if mean_only:
                    res = res.mean()

        else:
            # 使用回归器的情况
            res = mean_squared_error(y_true=y_train, y_pred=y_pred)

        if self.verbose > 2:
            report = classification_report(
                y_true=y_train, y_pred=y_pred
            )  # 训练集上,几乎总是是满分
            print(report, self.model.__class__.__name__)
        return res

    def validate_empty_array(self, X_test=[], y_test=[]):
        if len(X_test) == 0:
            raise ValueError("X is empty")
        if len(y_test) == 0:
            raise ValueError("y is empty")

    def meta_paths_of_db(self, db, partition="test"):
        res = meta_paths_of_db(
            db=db,
            e_config=self.e_config,
            change_type="str",
            partition=partition,
        )
        return res

    def update_test_set(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def update_test_set_by_meta(self, test_meta):
        """
        这个函数设计用来做跨库识别试验
        仅仅替换测试集为不同库,本身没有针对跨库进行优化


        Load test data from given test metadata file paths and update instance's test set attributes.

        Args:
            test_meta (list of str): List of file paths of test metadata files.

        Returns:
            None

        examples:
        >>> rec = EmotionRecognizer(model=my_model,**meta_dict, verbose=1)
        >>> rec.train()
        >>> rec.update_test_set_by_meta(test_emodb_csv)
        >>> rec.test_meta_files
        >>> 'meta_files\\test_emodb_HNS.csv'
        >>> rec.X_test.shape
        >>> (43,180)

        >>> rec.test_score()
        >>> 0.4651

        """
        # rec.update_test_set_by_meta(test_emodb_csv)
        self.test_meta_files = test_meta
        print(test_meta, "@{test_meta}")
        test_data = load_data_from_meta(
            test_meta_files=test_meta, e_config=self.e_config, f_config=self.f_config
        )

        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        # 设置实例的各个属性
        self.test_audio_paths = test_data["test_audio_paths"]
        self.update_test_set(X_test, y_test)

    def train_score(self, X_train=None, y_train=None):
        """
        Calculates accuracy score on training data
        if `self.classification` is True, the metric used is accuracy,
        Mean-Squared-Error is used otherwise (regression)
        """
        if X_train is None or y_train is None:
            X_train = self.X_train
            y_train = self.y_train
        y_pred = self.model.predict(X_train)
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
            matrix_df = pd.DataFrame(
                matrix,
                index=[f"true_{e}" for e in self.e_config],
                columns=[f"predicted_{e}" for e in self.e_config],
            )
        return matrix_df

    def draw_confusion_matrix(self):
        """Calculates the confusion matrix and shows it"""
        matrix = self.confusion_matrix(percentage=False, labeled=False)
        # TODO: add labels, title, legends, etc.
        pl.imshow(matrix, cmap="binary")
        pl.show()

    def count_samples_in_partition(self, emotion, partition):
        """
        Get the number of data samples of the `emotion` class in a particular `partition` ('test' or 'train').

        :param emotion: The emotion class to count.
        :param partition: The partition to count samples in ('test' or 'train').
        :return: The number of data samples of the `emotion` class in the `partition`.
        :raises ValueError: If `y_test` or `y_train` is `None`.
        """
        partition = validate_partition(partition, Noneable=False)
        if partition == "test":
            if self.y_test is None:
                raise ValueError("y_test is None")
            count = sum(1 for y in self.y_test if y == emotion)
        else:
            if self.y_train is None:
                raise ValueError("y_train is None")
            count = sum(1 for y in self.y_train if y == emotion)
        return count

    def count_samples_by_class(self):
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
            n_train = self.count_samples_in_partition(emotion, "train")
            n_test = self.count_samples_in_partition(emotion, "test")
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

    def get_random_emotion_index(self, emotion, partition="train"):
        """
        Returns a random index of a `partition` sample with the given `emotion`.

        Args:
            emotion (str): The name of the emotion to look for.
            partition (str): The partition to sample from. Only "train" or "test" are accepted.

        Returns:
            int: The index of a random sample with the given `emotion` in the specified `partition`.

        Raises:
            TypeError: If `partition` is not "train" or "test".
        """
        partition = validate_partition(partition, Noneable=False)
        indices = []
        if partition == "train":
            indices = [i for i, y in enumerate(self.y_train) if y == emotion]
        elif partition == "test":
            indices = [i for i, y in enumerate(self.y_test) if y == emotion]

        return random.choice(indices)


from sklearn.datasets import load_iris, load_digits
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier


# def get_stacking_clf():
#     random_state = 42
#     # 定义初级学习器
#     estimators_basic_mix = estrs_basic_mix( )
#     estimators_dt = estrs_esdt(random_state)
#     estimators_basic_dt_mix = estrs_basic_dt_mix( )
#     estimators_basic_dt_mpl_mix = estrs_basic_dt_mpl_mix(
#         random_state, estimators_basic_dt_mix
#     )
#     estimators_basic_dmb_mix = estimators_basic_dt_mix + [("gnb", GaussianNB())]
#     # 简单堆叠
#     stack_dt_linear = StackingClassifier(
#         estimators=estimators_dt, final_estimator=LogisticRegression()
#     )
#     stack_mix_linear = StackingClassifier(
#         estimators=estimators_basic_mix, final_estimator=LogisticRegression()
#     )
#     # 收敛困难(效果不如mix_linear)
#     stack_mix_svm = StackingClassifier(
#         estimators=estimators_basic_mix,
#         final_estimator=SVC(),
#     )
#     stack_basic_dt_mix = StackingClassifier(
#         estimators=estimators_basic_dt_mix, final_estimator=LogisticRegression()
#     )
#     # 数据量太小
#     stack_basic_dt_mlp_mix = StackingClassifier(
#         estimators=estimators_basic_dt_mpl_mix, final_estimator=LogisticRegression()
#     )
#     stack_dmb_mix = StackingClassifier(
#         estimators=estimators_basic_dmb_mix,
#         final_estimator=LogisticRegression(),
#     )
#     # stack1 = StackingClassifier(
#     #     estimators=[("gbc", GaussianNB())],
#     #     final_estimator=LogisticRegression(),
#     # )
#     # 多层堆叠
#     ##定义最后一层
#     stack_final_layer = StackingClassifier(
#         estimators=[("gbc", GradientBoostingClassifier()), ("svc", SVC())],
#         final_estimator=LogisticRegression(),
#     )
#     ##堆叠二层(容易过拟合)
#     stack_multilayer = StackingClassifier(
#         estimators=estimators_dt, final_estimator=stack_final_layer
#     )
#     stack = StackingClassifier(estimators=estimators_dt, final_estimator=SVC())

#     # return stack_dt_linear
#     # return stack_mix_linear
#     # return stack_mix_svm
#     # return stack_basic_dt_mix
#     return stack_dmb_mix


def estrs_basic_mlp_mix():
    estimators_basic_dt_mpl_mix = estrs_basic_mix() + [
        (
            "mpl",
            MLPClassifier(
                alpha=0.01,
                batch_size=512,
                hidden_layer_sizes=(300,),
                learning_rate="adaptive",
                max_iter=400,
                random_state=random_state,
            ),
        )
    ]

    return estimators_basic_dt_mpl_mix


def estrs_esdt():
    estimators_dt = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=10, max_depth=3, random_state=random_state
            ),
        ),
        (
            "adab",
            AdaBoostClassifier(
                n_estimators=10, learning_rate=0.1, random_state=random_state
            ),
        ),
        ("gradb", GradientBoostingClassifier()),
    ]

    return estimators_dt


def estrs_basic_esdt_mix():
    res = estrs_basic_mix() + estrs_esdt()
    return res


def estrs_simple():
    res = [
        ("knc", KNeighborsClassifier(n_neighbors=3, p=1, weights="distance")),
        ("gnb", GaussianNB()),
    ]
    return res


def estrs_basic_mix():
    """包含常用的个体学习器,可以作为Stacking的第一层
    还可以作为更加复杂的第一层的基础部分,采用列表相加的方式进一步扩充
    例如添加集成学习(随机森林,梯度提升等)

    具体包括以下模型:

    - 线性模型(lsvr,rdcv,logistic)
    - k近邻(knc)
    - 贝叶斯决策(gnb)
    - 决策树(dt)

    这些个体学习器较为多样,理论上有利于提高集成学习的泛化能力

    Returns
    -------
    list[estimator]
        基础个体学习器列表
    """
    estimators_basic_mix = [
        # ("svc",(SVC(C=10, gamma=0.001,random_state=random_state))),
        # ("lsvr", make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))),
        ("lsvr", (LinearSVC(max_iter=5000, random_state=random_state))),
        ("rdcv", RidgeClassifierCV()),
        ("logistic", LogisticRegression()),
        ("knc", KNeighborsClassifier(n_neighbors=3, p=1, weights="distance")),
        ("gnb", GaussianNB()),
        (
            "dtc",
            DecisionTreeClassifier(
                criterion="entropy", max_depth=7, max_features="sqrt"
            ),
        ),
    ]

    return estimators_basic_mix


def get_clfs():
    """需要评估和分析对比的估计器
    svc = SVC(C=0.001, gamma=0.001, kernel="poly", probability=True)
    knn=KNeighborsClassifier(n_neighbors=3, p=1, weights='distance')
    Best for DecisionTreeClassifier: {'criterion': 'entropy', 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
    dt = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=7,
        max_features=None,
        min_samples_leaf=1,
        min_samples_split=2
    )

    Returns
    -------
    list[estimator]
        sklearn.estimator估计器列表
    """

    rfc = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    lsvc = LinearSVC(random_state=random_state)
    svc = SVC()
    mlp = MLPClassifier()
    rdcv = RidgeClassifierCV()
    gnb = GaussianNB()
    plsvc = make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))
    adab = AdaBoostClassifier()
    gbc = GradientBoostingClassifier()

    # stack = get_stacking_clf()
    stack_basic_svc = StackingClassifier(
        estimators=estrs_basic_mix(), final_estimator=SVC()
    )
    stack_basic_rf = StackingClassifier(
        estimators=estrs_basic_mix(), final_estimator=RandomForestClassifier()
    )
    stack_basic_mlp = StackingClassifier(
        estimators=estrs_basic_mix(), final_estimator=MLPClassifier()
    )
    # estrs_basic_esdt_mix作为第一层
    stack_basic_esdt_gnb = StackingClassifier(
        estimators=estrs_basic_esdt_mix(), final_estimator=GaussianNB()
    )
    # estrs_basic_mlp_mix作为第一层

    stack_basic_mlp_gnb = StackingClassifier(
        estimators=estrs_basic_mlp_mix(), final_estimator=GaussianNB()
    )
    stack_basic_mlp_lr = StackingClassifier(
        estimators=estrs_basic_mlp_mix(), final_estimator=LogisticRegression()
    )
    stack_simple = StackingClassifier(
        estimators=estrs_simple(), final_estimator=LinearSVC()
    )
    stack_multilayer = StackingClassifier(
        estimators=estrs_basic_esdt_mix(), final_estimator=stack_simple
    )
    # 配置待评估的分类器的列表
    clfs = [
        # 堆叠泛化
        stack_basic_svc,
        stack_basic_rf,
        stack_basic_esdt_gnb,
        stack_simple,
        # 计算量大:
        # stack_basic_mlp,
        # stack_basic_mlp_gnb,
        # stack_basic_mlp_lr,
        # stack_multilayer,
        # 基础分类器
        # gnb,
        # rfc,
        # rdcv,
        # dt,
        # lsvc,
        # svc,
        # mlp,
        # plsvc,
        # adab,
        # gbc,
        None,  # None表示自动计算最优模型(但是范围限制在grid中有定义的那一部分.)
    ]
    return clfs


def main():
    clfs = get_clfs()

    passive_emo = ["angry", "sad"]
    passive_emo_others = passive_emo + ["others"]
    typical_emo = [
        "happy",
        "neutral",
        "sad",
    ]
    AHS = ["angry", "happy", "sad"]
    AHSO = ["angry", "neutral", "sad", "others"]

    AHNS = ["angry", "happy", "neutral", "sad"]
    # e_config = typical_emo
    f_config = ["mfcc"]
    e_config = e_config_def
    # f_config = f_config_def
    # 配置语料库
    ## 同库实验
    meta_dict = mp.get_single_db_pair_dict(emodb)
    ## 跨库实验
    # meta_dict = mp.emodb_savee
    # meta_dict = mp.ravdess_savee
    meta_dict = mp.emodb_ravdess

    res_list = []
    # 评估定义在clfs中的模型的性能:
    for i, clf in enumerate(clfs):
        res_dict = assess_model(e_config, f_config, clf, meta_dict)
        res_list.append(res_dict)
        print(f"当前评估第{i+1}个模型")

    res_list.sort(key=lambda res_dict: res_dict["test_score"], reverse=True)
    for i, res_dict in enumerate(res_list):
        print(i + 1, "--" * 30, "\n")
        for key, value in res_dict.items():
            if key == "er":
                print(key, ":", value.model)
            elif key in ["report", "confusion_matrix"]:
                print(key, ":\n", value)
            else:
                print(key, ":", value)

        # print(model, "@{model}")
        # print(f"{train_score=}")
        # print(f"{test_score=}")
        # # 查看混淆矩阵
        # print(confusion_matrix)
        # # 查看辅助性能指标报告
        # print(report)


def assess_model(e_config, f_config, model, meta_dict):
    er = EmotionRecognizer(
        model=model,
        **meta_dict,
        e_config=e_config,
        f_config=f_config,
        balance=True,
        cross=True,  # 执行跨库任务,调整测试数据集读入
        verbose=0,
        # std_scaler=False,
        # pca_params=dict(n_components=39)
        # std_scaler=False,
        # pca={"n_components":"mle"}
        # pca={'n_components': 60}
    )
    # 显示调用训练方法(相当于调用sklearn.estimator.fit)
    er.train()
    # 评估模型的各项性能指标
    train_score = er.train_score()
    test_score = er.test_score()
    confusion_matrix = er.confusion_matrix()
    report = er.check_report()
    # cv_score = er.model_cv_score()

    # print(model, "@{model}")
    # print(f"{train_score=}")
    # print(f"{test_score=}")
    # # 查看混淆矩阵
    # print(confusion_matrix)
    # # 查看辅助性能指标报告
    # print(report)
    # 交叉验证得分
    # print(f"{cv_score=}")

    return dict(
        er=er,
        train_score=train_score,
        test_score=test_score,
        confusion_matrix=confusion_matrix,
        report=report,
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        er = main()
##

# file=r'D:\repos\CCSER\SER\data\savee\AudioData\DC\h01.wav'
# file=meta.speech_dbs_dir/emodb/r'wav/03a01Fa.wav'
# predict_res=er.predict(file)
# print(f"{predict_res=}")
# predict_proba=er.predict_proba(file)
# print(f"{predict_proba=}")

##

# data=rec.load_data()

# data = load_data(
#     train_meta_files=train_meta_files,
#     test_meta_files=test_meta_files,
#     balance=False,
# )

# X_train = data["X_train"]
# y_train = data["y_train"]
# rec.train(X_train=X_train, y_train=y_train)
