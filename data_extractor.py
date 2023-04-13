import os
import sys
from collections import defaultdict
from email.mime import audio
from functools import partial

import numpy as np
import pandas as pd
import tqdm

import EF
from create_csv import create_csv_by_metaname
from EF import AHNPS, HNS, AHNPS_dict, HNS_dict, MCM_dict, e_config_def, f_config_def
from MetaPath import (
    create_tag_name,
    features_dir,
    get_features_tag,
    get_first_letters,
    test_emodb_csv,
    test_ravdess_csv,
    train_emodb_csv,
    train_ravdess_csv,
    validate_partition,
)
from utils import extract_feature

# from pathlib import Path
Series = pd.Series
DataFrame = pd.DataFrame


class AudioExtractor:
    """A class that is used to featurize audio clips, and provide
    them to the machine learning algorithms for training and testing
    和特征提取不同,本模块负责处理特征提取之后的数据处理,特征提取(参看utils.extract_feature方法)
    本模块尝试从指定目录(默认从features)目录导入特征文件(.npy)(由numpy提供的对象存储方案)
    """

    def __init__(
        self,
        dbs=None,
        f_config=None,
        e_config=None,
        features_dir=features_dir,
        verbose=True,
        classification_task=True,
        balance=True,
    ):
        """

        Params:
        -
        audio_config (dict):
            the dictionary that indicates what features to extract from the audio file,
            default is {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': False, 'tonnetz': False}
            (i.e mfcc, chroma and mel)

        verbose (bool/int):
            verbosity level, False for silence, True for info, default is True

        features_folder_name (str):
            the folder to store output features extracted,
            default is "features".

        classification (bool):
            whether it is a classification or regression, default is True (i.e classification)

        emotions (list):
            list of emotions to be extracted, default is [ 'happy','neutral' ,'sad']

        balance (bool):
            whether to balance dataset (both training and testing), default is True
        """
        self.f_config = f_config if f_config else f_config_def
        self.e_config = e_config if e_config else e_config_def
        self.dbs = dbs
        self.verbose = verbose
        self.features_dir = features_dir  # 默认为features目录
        self.classification_task = classification_task
        self.balance = balance
        # input dimension
        self.feature_dimension = None
        # 记录最后一次提取语音文件信息
        self.audio_paths = []
        self.emotions = []

        # partition attributes
        self.train_audio_paths = []
        self.train_emotions = []
        self.train_features = []

        self.test_audio_paths = []
        self.test_emotions = []
        self.test_features = []
        # 使用字典打包

    def get_partition_features(self, partition) -> np.ndarray:
        """将包含若干个二维ndarray的列表vstack成1个二维ndarray

        Parameters
        ----------
        partition : str
            "train"|"test"

        Returns
        -------
        np.ndarray
            合并完成的矩阵

        Raises
        ------
        ValueError
            _description_
        """
        # print("len(self.train_features),len(self.test_features):")
        # print(len(self.train_features),len(self.test_features))
        # return
        partition=validate_partition(partition)
        if partition == "test":
            res = np.vstack(self.test_features) if self.test_features else np.array([])
        else :
            res = (
                np.vstack(self.train_features) if self.train_features else np.array([])
            )
      
        return res

    def load_metadata(self, meta_file):
        """
        从meta_files(文件)中读取语料库各条语音的信息;

        Read metadata from a  file & Extract and loads features of audio files

        Parameters
        ----------
        meta_files : list[str]|str
            需要读取的meta文件

        """
        # empty dataframe
        df = pd.DataFrame({"path": [], "emotion": []})
        # 合并所有需要读入的文件
        # for meta_file in meta_files:
        #     # concat dataframes
        #     df = pd.concat((df, pd.read_csv(meta_file)), sort=False)
        if not os.path.exists(meta_file):
            # create_csv_by_meta_name
            create_csv_by_metaname(meta_file)

        df = pd.read_csv(meta_file)
        if self.verbose:
            print("[Info] Loading audio file paths and its corresponding labels...")
        # get columns
        # 抽取音频文件路径信息和对应的情感标签
        # audio_paths_:list
        # emotions_:list
        audio_paths, emotions = list(df["path"]), list(df["emotion"])
        return audio_paths, emotions

    def _class2rgr(self):
        # 如果执行回归预测,那么本模块将增加self.categories_rgr将标签数值化
        # 默认情况下,系统执行分类预测,标签是情感名词字符串
        # 数值化类别标签,便于使用回归算法预测(设置3情感和5情感2种模式)
        # if not classification, convert emotions to numbers
        if not self.classification_task and self.e_config:
            if len(self.e_config) == 3:
                # HNS情感模式(Happy,Neutral,Sad)
                self.categories_rgr = HNS_dict
            elif len(self.e_config) == 5:
                # AHNPS情感模式:Angry,Happy,Neutral,PleasantSuprise,Sad
                self.categories_rgr = AHNPS_dict
            else:
                raise TypeError(f"Regression is only for either {HNS} or {AHNPS}")
            # TODO (improve)
            emotions = self.e_config  # 可以再优化robustness
            emotions = [self.categories_rgr[e] for e in emotions]  # 字符串标签数字化

    def _update_partition_attributes(
        self,
        partition,
        audio_paths=None,
        emotions=None,
        features=None,
    ):
        """
         用于实现增量提取
         附加属性集设置和增量
         初次调用会设置新属性

        ## Note1:
        警惕numpy的ndarray数组判断bool值时的错误做法:
        `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`
        这个错误的是由于ndarray对象在bool上下文(比如if a)(a标识ndarray对象)
        的语句中,由于ndarray包含的size个元素各自有自己的`True`/`False`值,需要用any()或all()来消除歧义
        不过有时候我们仅仅想知道被判断的对象是否为非空或非None,那么可以用a.szie属性来判断元素数量,用`a is None`

        ## Note2:
        print(id(self.test_features), "@{Id@self.test_features}")
        print(id(features_attr), "@{Id@features_attr}")
        print(id(features),"@{Id@features}")


        Parameters
        ----------
        partition : str
            "train|test"
        audio_paths : list[str]
            train/test set file path
        emotions : list[str]
            tr/te set file path
        features : ndarray, optional
            由特征提取环节返回的特征数组, by default None

        """
        # if(audio_paths_ and emotions_):
        # audio_paths = self.audio_paths
        # emotions = self.e_config
        np.mean([12, 3])
        # 设置训练集属性
        partition=validate_partition(partition)
        verbose=self.verbose
        if verbose:
            print(f"[Info] Adding  {partition} samples")

        # partition_attribute_set
        audio_paths_attr, emotions_attr, features_attr = self.partition_attributes_set(
            partition
        )
        if audio_paths:
            audio_paths_attr += audio_paths
        if emotions:
            emotions_attr += emotions

        # 这里要区别对待features,原因参考python的id()方法
        if features is not None:
            features_attr += [features]  # 将features打包为列表在并入
            # np.vstack(features)
        if verbose>=2:
            #检查训练集和测试集属性情况
            check_1=self.partition_attributes_set(partition)
            for attr in check_1:
                nd_attr=np.array(attr,dtype=object)
                print(nd_attr.shape,id(attr))
            


    def partition_attribute_update_handler(
        self, partition, features, audio_paths, emotions
    ):
        pass

    # @deprecate()
    def partition_attributes_set(self, partition=None,verbose=1):
        """
        返回初始值为空列表的partition属性
        - !要求调用本方法返回的变量必须使用+=的方法修改,否则达不到预期效果
        - 对于features属性,在访问的时候将是不同的特征矩阵
        - 其他写法
        if partition == "train":

            self.train_audio_paths+=value

        elif partition == "test":
            self.test_audio_paths+=value

        Parameters
        ----------
        partition : str
            _description_
        value : _type_
            _description_
        """
        partition=validate_partition(partition=partition)

        attributes = []
        if partition == "train":
            attributes = [
                self.train_audio_paths,
                self.train_emotions,
                self.train_features,
            ]
        elif partition == "test":
            attributes = [self.test_audio_paths, self.test_emotions, self.test_features]
        # print(attributes,"@{attributes}:",partition)
        if verbose:
            print("ids of attributes set:")
            print([id(attr) for attr in attributes])
        return attributes

    def _extract_feature_in_meta(self, partition=None, meta_file=None):
        """根据meta_files提取相应语音文件的特征
        这里仅完成单次提取

        Parameters
        ----------
        meta_files : list[str]|str
            meta_files
        partition : str
            标记被提取文件是来自训练集还是测试集(验证集)
        """
        audio_paths, emotions = self.load_metadata(meta_file)
        # 将计算结果保存为对象属性

        self.audio_paths = audio_paths
        self.emotions = emotions

        # make features folder if does not exist
        if not os.path.isdir(self.features_dir):
            os.mkdir(self.features_dir)

        # construct features file name

        n_samples = len(audio_paths)  # 计算要处理的音频文件数

        features_file_name = create_tag_name(
            partition=partition,
            # db=db,
            e_config=self.e_config,
            f_config=self.f_config,
            n_samples=n_samples,
            ext="npy",
        )
        # 构造保存特征矩阵npy文件的路径
        features_file_path = os.path.join(
            self.features_dir,
            features_file_name,
        )
        if os.path.isfile(features_file_path):
            # if file already exists, just load
            if self.verbose:
                print("文件特征矩阵文件已经存在,直接导入:loading...")
            features = np.load(features_file_path)
        else:
            # file does not exist, extract those features and dump them into the file
            if self.verbose:
                print("npy文件不存在,尝试创建...")
            # 如果尚未提取过特征,则在此处进行提取,同时保存提取结果,以便下次直接使用
            features = self.features_save(partition, audio_paths, features_file_path)

        return features, audio_paths, emotions

    def features_save(self, partition, audio_paths, features_file_path):
        """将提取的特征(ndarray)保存持久化保存(为npy文件)
        利用qtmd提供可视化特征抽取进度

        Parameters
        ----------
        partition : str
            "test|train"
        audio_paths_ : str
            音频文件的路径
        features_file_path : str
            保存文件名

        Returns
        -------
        ndarray
            提取的特征数组
        """
        features = []
        # print(audio_paths)
        # append = features.append
        # 特征提取是一个比较耗时的过程,特征种类越多越耗时,这里采用tqdm显示特征提取进度条(已处理文件/文件总数)
        for audio_file in tqdm.tqdm(
            audio_paths, f"Extracting features for {partition}"
        ):
            # 调用utils模块中的extract_featrue进行特征提取
            f_config = self.f_config
            feature = extract_feature(audio_file, f_config=f_config)
            # 默认设置input_dimension为feature.shape[0]
            if self.feature_dimension is None:
                self.feature_dimension = feature.shape[0]
            # 把当前文件提取出来特征添加到features数组中
            features.append(feature)
        # 此时所有文件特征提取完毕,将其用numpy保存为npy文件
        features = np.array(features)  # 构成二维数组
        np.save(features_file_path, features)
        print(features)
        # sys.exit(1)
        return features

    def extract_update(self, partition, meta_files, verbose=1):
        """特征提取和self属性更新维护
        多次调用将执行增量提取,根据partition的取值,将每次的提取结果增量更新self的相应属性集
        Parameters
        ----------
        partition : str
            "train"|"test"
        meta_files : list[str]|str
            _description_
        """
        if meta_files is None:
            return meta_files
        if isinstance(meta_files, str):
            print(f"cast the {meta_files} str to [str]")
            meta_files = [meta_files]

        # 执行特征提取
        for meta_file in meta_files:
            features, audio_paths, emotions = self._extract_feature_in_meta(
                meta_file=meta_file, partition=partition
            )
            if verbose >= 1:
                print(features.shape, "@{feature.shape}")
            if verbose >= 2:
                print(features, "@{features}")
            # 每根据一个meta_file提取一批特征,就更新到self的相关属性集中
            self._update_partition_attributes(
                partition=partition,
                audio_paths=audio_paths,
                emotions=emotions,
                features=features,
            )

    def load_data_preprocessing(self, meta_files, partition=None, shuffle=False):
        """将特征提取和属性设置以及打乱和平衡操作打包处理
        AE对象在如数据集后可选的数据处理操作(balance&shuffle)

        Parameters
        ----------
        meta_files : list[str]|str
            需要载入数据的meta信息
        partition : str, optional
            "test|train", by default None
        shuffle : bool, optional
            是否执行打乱数据顺序操作, by default False
        """
        # self.load_metadata_from_desc_file(meta_files)
        print(meta_files)
        # sys.exit()
        self.extract_update(partition=partition, meta_files=meta_files)

        # balancing the datasets ( both training or testing )
        if self.balance:
            self._balance_data(partition=partition)
        # shuffle
        if shuffle:
            self.shuffle_data_by_partition(partition)



    def shuffle_data_by_partition(self, partition):
        """打乱数据顺序

        Parameters
        ----------
        partition : str
            "train"|"test"

        Raises
        ------
        TypeError

        """
        if partition == "train":
            (
                self.train_audio_paths,
                self.train_emotions,
                self.train_features,
            ) = shuffle_data(
                self.train_audio_paths, self.train_emotions, self.train_features
            )
        elif partition == "test":
            (
                self.test_audio_paths,
                self.test_emotions,
                self.test_features,
            ) = shuffle_data(
                self.test_audio_paths, self.test_emotions, self.test_features
            )
        else:
            raise TypeError("Invalid partition, must be either train/test")

    def _balance_data(self, partition=None):
        """
        对训练集/测试集的数据做平衡处理

        """
        partition=validate_partition(partition=partition)

        audio_paths, counter, emotions_tags, features = self.emotions_counter(partition)

        # get the minimum data samples to balance to
        minimum = self.validate_balance_task(counter)
        if minimum == 0:
            return

        # 构造并初始化{情感:数量}字典
        counter = self.count_dict()

        dd = self.balanced_dict(audio_paths, counter, emotions_tags, features, minimum)

        # 将类别平衡处理好的元组形式重新解析回基本python类型对象
        audio_paths, emotions_tags, features = self.parse_balanced_data(dd)
        # 将解析结果更新回self对象的相应属性上
        self.update_balanced_attributes(partition, audio_paths, emotions_tags, features)

    def validate_balance_task(self, counter):
        minimum = min(counter)
        if self.verbose:
            print("[*] Balancing the dataset to the minimum value:", minimum)
        if minimum == 0:
            # won't balance, otherwise 0 samples will be loaded
            print("[!] One class has 0 samples, setting balance to False")
            self.balance = False
        return minimum

    def balanced_dict(self, audio_paths, counter, emotions_tags, features, minimum):
        """构造平衡处理好数据集的字典

        实现说明:本方法主要借助defaultdict实现,简称dd
        它是 Python 中的一个字典子类，它在字典的基础上添加了一个默认工厂函数，
        使得在访问字典中不存在的键时，可以返回一个默认值而不是引发 KeyError 异常。
        defaultdict 的构造函数需要一个参数，即默认工厂函数。
        默认工厂函数可以是 Python 内置类型（如 int、list、set 等），也可以是用户自定义函数。
        当访问字典中不存在的键时，如果使用了默认工厂函数，则会自动创建一个新的键，
        并将其对应的值初始化为默认值（由默认工厂函数返回）。

        在dd的帮助下,我们可以轻松的统计(情感)类别数未知的情况下,统计各个情感的文件数

        Parameters
        ----------
        audio_paths : list
            _description_
        counter :
            _description_
        emotions_tags : _type_
            _description_
        features : _type_
            _description_
        minimum : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        dd = defaultdict(list)  #
        for e, feature, audio_path in zip(emotions_tags, features, audio_paths):
            if counter[e] >= minimum:
                # minimum value exceeded
                continue
            counter[e] += 1
            dd[e].append((feature, audio_path))
        # data={}
        # 临时属性用于notebook中调试
        self.dd_debug = dd
        # print(dd)
        return dd

    def count_dict(self,verbose=0):
        """构造并初始化{情感:数量}字典
        Returns
        -------
        dict
            用于统计平衡数据的字典
        """
        if self.classification_task:
            res = {e: 0 for e in self.e_config}  # type:ignore
        else:
            res = {e: 0 for e in self.categories_rgr.values()}

        if verbose:
            print("{res}:")
            print(res)
        return res

    def parse_balanced_data(self, dd):
        emotions_tags, features, audio_paths = [[] for _ in range(3)]
        for emo, f_ap in dd.items():
            for feature, audio_path in f_ap:
                emotions_tags.append(emo)
                features.append(feature)
                audio_paths.append(audio_path)
        return audio_paths, emotions_tags, features

    def update_balanced_attributes(
        self, partition, audio_paths, emotions_tags, features
    ):
        """将解析结果更新回self对象的相应属性上

        Parameters
        ----------
        partition : str
            "test|train"
        audio_paths : list
            路径
        emotions_tags : list
            情感标签
        features : list
            特征

        Raises
        ------
        TypeError
            数据集目标名字划分非法(test|train)
        """
        partition=validate_partition(partition=partition,Noneable=False)
        if partition == "train":
            self.train_emotions = emotions_tags
            self.train_features = features
            self.train_audio_paths = audio_paths
        elif partition == "test":
            self.test_emotions = emotions_tags
            self.test_features = features
            self.test_audio_paths = audio_paths


    def emotions_counter(self, partition):
        # 统一变量名
        data = [None] * 3
        if partition == "train":
            data = (self.train_emotions, self.train_features, self.train_audio_paths)
        elif partition == "test":
            data = (self.test_emotions, self.test_features, self.test_audio_paths)
        else:
            raise TypeError("Invalid partition, must be either train/test")
        emotions_tags, features, audio_paths = data
        # 为了使各中情感的文件数量相等,分别统计不同情感的文件数
        counter = []  # count中的元素个数等于len(self.emotions)
        # 分类预测
        if self.classification_task:
            # 遍历需要抽取的情感标签
            e_config = self.e_config if self.e_config else e_config_def
            for emo in e_config:
                n_samples_of_emo = len([e for e in emotions_tags if e == emo])
                counter.append(n_samples_of_emo)
        # 回归预测
        else:
            # regression, take actual numbers, not label emotion
            for emo in self.categories_rgr.values():
                counter.append(len([e for e in emotions_tags if e == emo]))
        return audio_paths, counter, emotions_tags, features
        # return counter


def shuffle_data(audio_paths, emotions, features):
    """Shuffle the data
        (called after making a complete pass through
        training or validation data during the training process)

    Params:
    -
        audio_paths (list): Paths to audio clips
        emotions (list): Emotions in each audio clip
        features (list): features audio clips
    """
    length = len(audio_paths)
    # print(length)
    # print([len(item) for item in (audio_paths,emotions,features)])
    if length:
        # 对range(length)的乱序排列列表
        # 根据统一的乱序序列,便于统一audio_paths,emotions,features
        # 因此这里不可用直接地对三个列表各自地运行shuffle或permutation,会导致对应不上
        p = np.random.permutation(length)
        audio_paths = [audio_paths[i] for i in p]
        emotions = [emotions[i] for i in p]
        features = [features[i] for i in p]

    return audio_paths, emotions, features


def load_data(
    train_meta_files=None,
    test_meta_files=None,
    f_config=f_config_def,
    e_config=e_config_def,
    classification_task=True,
    shuffle=False,
    balance=False,
) -> dict:
    """导入语音数据,并返回numpy打包train/test dataset相关属性的ndarray类型

    Parameters
    ----------
    train_desc_files : list
        train_meta_files
    test_desc_files : list
        test_meta_files`
    f_config : dict, optional
        需要提取的特征, by default None
    e_config : list, optional
        需要使用的情感类别字符串构成的列表, by default ['sad', 'neutral', 'happy']
    classification_task : bool, optional
        是否采用分类器(否则使用回归模型), by default True
    shuffle : bool, optional
        是否打乱顺序, by default True
    balance : bool, optional
        是否进行数据平衡, by default True

    Returns
    -------
    dict
        返回载入情感特征文件的矩阵构成的字典
    """
    # instantiate the class(实例化一个AudioExtractor实例)
    ae = AudioExtractor(
        f_config=f_config,
        e_config=e_config,
        classification_task=classification_task,
        balance=balance,
        verbose=True,
    )
    # Loads training data
    ae.load_data_preprocessing(train_meta_files, partition="train", shuffle=shuffle)
    # Loads testing data
    ae.load_data_preprocessing(test_meta_files, partition="test", shuffle=shuffle)

    #以train集为例检查self属性
    # print("ae.train_audio_paths:\n", ae.train_audio_paths)
    # X_train, X_test, y_train, y_test
    
    return {
        # "X_train": np.array(ae.train_features),
        # "X_test": np.array(ae.test_features),
        "X_train": ae.get_partition_features("train"),
        "X_test": ae.get_partition_features("test"),
        "y_train": np.array(ae.train_emotions),
        "y_test": np.array(ae.test_emotions),
        "train_audio_paths": ae.train_audio_paths,
        "test_audio_paths": ae.test_audio_paths,
        "balance": ae.balance,
        "ae": ae,
    }


if __name__ == "__main__":

    # ae = AudioExtractor(f_config=f_config_def)
    # print(ae)

    data = load_data(
        # train_meta_files=train_emodb_csv,
        test_meta_files=test_emodb_csv,
        f_config=f_config_def,
        # balance=False,
        balance=True,
    )
