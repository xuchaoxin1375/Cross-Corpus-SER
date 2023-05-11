##

import os
import sys
from collections import defaultdict
from pathlib import Path

import ipdb
import numpy as np
import pandas as pd
import tqdm
from joblib import load

import config.MetaPath as mp
from audio.core import extract_feature_of_audio
from audio.create_meta import create_csv_by_metaname
from config.EF import (AHNPS, HNS, AHNPS_dict, HNS_dict, e_config_def,
                       f_config_def)
from config.MetaPath import (ava_dbs, ava_fts_params, create_tag_name, emodb,
                             features_dir, get_first_letters, meta_dir,
                             project_dir, train_emodb_csv, validate_partition)

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
        shuffle=True,
        feature_transforms_dict=None,
    ):
        """
        初始化AE对象,在init中对构造器中传入None或者不传值得参数设置了默认值,默认参数为None是参考Numpy的风格
        然而默认值设置在init也有不好的地方,比如这容易出现一些默认但是出乎意料的行为;所以应该在注释部分尽可能地详细说明

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
        self.feature_transforms = feature_transforms_dict
        
        self.balance = balance
        self.shuffle = shuffle
        # input dimension
        self.feature_dimension = None
        self.feature_dimension_pca = None
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
        self.pca = None
    def pathlike_to_list(self, meta_paths):
        if isinstance(meta_paths, str) or isinstance(meta_paths,Path):
            # print(f"cast the '{meta_paths}' to [str]")
            meta_paths = [meta_paths]
        return meta_paths

    def get_partition_features(self, partition) -> np.ndarray:
        """将包含若干个二维ndarray的列表vstack成1个二维ndarray
        self.features是一个包含若干个同列数的二维数组的list,这里将list中的二维数组合并为1个二维数组返回之
        Parameters
        ----------
        partition : str
            "train"|"test"

        Returns
        -------
        np.ndarray
            合并完成的二维数组,而不是list

        Raises
        ------
        ValueError
            _description_
        """
        # print("len(self.train_features),len(self.test_features):")
        # print(len(self.train_features),len(self.test_features))
        # return
        partition = validate_partition(partition, Noneable=False)
        if partition == "test":
            res = np.vstack(self.test_features) if self.test_features else np.array([])
        else:
            res = (
                np.vstack(self.train_features) if self.train_features else np.array([])
            )

        return res

    def load_metadata(self, meta_files):
        """
        从给定meta_files(文件)路径中读取语料库各条语音的信息;
        如果需要读取的meta_files不存在,那么尝试解析meta_files(如果meta_files参数是一个符合可解析规范的字符串)
        这种情况下会调用create_meta模块中的create_csv_by_metaname函数进行meta文件构造

        Read metadata from a  file & Extract meta if according meta_files and loads features of audio files

        Parameters
        ----------
        meta_files : list[str]|str
            需要读取的meta文件
        Return
        -
        从meta中读取的信息:包括各语音文件的路径和情感标签

        """
        # empty dataframe
        df = pd.DataFrame({"path": [], "emotion": []})
        # 合并所有需要读入的文件
        # for meta_file in meta_files:
        #     # concat dataframes
        #     df = pd.concat((df, pd.read_csv(meta_file)), sort=False)
        if isinstance(meta_files, str):
            meta_files = [meta_files]
        if self.verbose:
            print("[I] Loading audio file paths and its corresponding labels...")
        # print("meta_files:", meta_files)

        # print("type(meta_files)", type(meta_files))

        meta_files = self.pathlike_to_list(meta_files)
        for meta_file in meta_files:
            if not os.path.exists(meta_file):
                # create_csv_by_meta_name
                print(f"{meta_file} does not exist,creating...😂")

                create_csv_by_metaname(meta_file, shuffle=self.shuffle)
            else:
                print(f"meta_file存在{meta_file}文件!")
            df_meta = pd.read_csv(meta_file)
            df = pd.concat((df, df_meta), sort=False)
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
        partition = validate_partition(partition)
        verbose = self.verbose
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
        if verbose >= 2:
            # 检查训练集和测试集属性情况
            check_1 = self.partition_attributes_set(partition)
            for attr in check_1:
                nd_attr = np.array(attr, dtype=object)
                # print(nd_attr.shape, id(attr))

    def partition_attribute_update_handler(
        self, partition, features, audio_paths, emotions
    ):
        pass

    # @deprecate()
    def partition_attributes_set(self, partition="", verbose=1):
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
        partition = validate_partition(partition=partition)

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
        if verbose >= 2:
            print("ids of attributes set:")
            print([id(attr) for attr in attributes])
        return attributes

    def _extract_feature_in_meta(self, partition="", meta_path="", verbose=1):
        """根据meta_files提取相应语音文件的特征
        这里仅完成单次提取

        矩阵文件名中的e_config字段暂定为self.e_config
        如果是这样,可能会和meta_path文件中的情感字段出现不一致的情况.

        Parameters
        ----------
        meta_files : list[str]|str
            meta_files
        partition : str
            标记被提取文件是来自训练集还是测试集(验证集)
        """
        # 检查数据集是否按照配置的情感进行筛选和划分:

        audio_paths, emotions = self.load_metadata(meta_path)
        # 将计算结果保存为对象属性
        self.audio_paths = audio_paths
        self.emotions = emotions

        # 尝试计算语料库的名字和情感配置名字
        db = self.fields_parse(meta_path)
        # 特征保存的目录检查(不存在则创建之)
        if not os.path.isdir(self.features_dir):
            os.mkdir(self.features_dir)

        n_samples = len(audio_paths)  # 计算要处理的音频文件数

        features_file_path = self.get_features_file_path(partition, db, n_samples,ext="npy")
        fts_file_path=self.get_features_file_path(partition, db, n_samples,ext="fts")
        if verbose:
            print(f"检查特征文件{features_file_path}是否存在...")
            print(f"{self.e_config=}")
            print(f"{self.f_config=}")

        ffp = os.path.isfile(features_file_path)
        # ftfp=os.path.isfile(fts_file_path)


        if ffp:
            # if file already exists, just load
            if self.verbose:
                print(f"特征矩阵文件(.npy)已经存在,直接导入:loading...")
            features = np.load(features_file_path)
            self.feature_dimension = features.shape[1]

        else:
            # file does not exist, extract those features and dump them into the file
            if self.verbose:
                print("npy文件不存在,尝试创建...")
            # 如果尚未提取过特征,则在此处进行提取,同时保存提取结果,以便下次直接使用
            # todo :目前对于特征标准化std_scaler和pca降维等操作,在将音频文件提取初始特征并执行transform的时候依赖于
            # 之前拟合号的transformer,如果要保留这类处理后的特征,
            # 为了能够在导入后能够提取新的音频做同样的降维操作就需要之前的transformer
            # 由于时间仓促,暂时不保存这类特征,而仅保存(缓存)初始特征
            save_obj = not self.feature_transforms
            print(self.feature_transforms,"@{self.feature_transforms}🎈")
            print(save_obj,"@{save_obj}")
            features = self.features_extract_save(
                partition, audio_paths, features_file_path,save_obj
            )
        # if ftfp:
        #     if self.verbose:
        #         print("fts文件(.fts)存在,直接导入...")
        #         # 导入到对象属性
        #         self.fts=load(fts_file_path)
        # else:
        #     if self.verbose:
        #         print("fts文件不存在,尝试创建...")
        #     print(f"请删除相关缓存文件{features_file_path}后重试...")

        return features, audio_paths, emotions

    def get_features_file_path(self, partition, db, n_samples,ext=""):
        fts=self.feature_transforms

        if fts is None:
            self.feature_transforms = {}
        features_file_name = create_tag_name(
            db=db,
            partition=partition,  # 建议保存特征文件时,这个字段置空即可
            e_config=self.e_config,
            f_config=self.f_config,
            n_samples=n_samples,
            ext=ext,
            **(self.feature_transforms),
        )

        # 构造保存特征矩阵npy文件的路径
        features_file_path = os.path.join(
            self.features_dir,
            features_file_name,
        )
        
        return features_file_path

    def fields_parse(self, meta_path):
        # 计算语料库字段名
        meta_fields, db = self.db_field_parse(meta_path)

        # 计算情感字段并检查
        self.validate_emotion_config_consistence(meta_fields)

        return db

    def db_field_parse(self, meta_path):
        meta_name = os.path.basename(meta_path)
        meta_name, ext = os.path.splitext(meta_name)
        meta_fields = meta_name.split("_")
        db = meta_fields[1]
        # print(f"{meta_path=}@")
        # print(f"{db=}@")

        db = db if db in ava_dbs else ""
        return meta_fields, db

    def validate_emotion_config_consistence(self, meta_fields):
        emotions_first_letters = meta_fields[-1]
        origin_efls = get_first_letters(self.e_config)
        # 检查情感配置是否具有一致性
        if emotions_first_letters != origin_efls:
            raise ValueError(
                f"{emotions_first_letters} is not inconsistant with {self.e_config}"
            )

    def features_extract_save(
        self, partition, audio_paths, features_file_path,save_obj=True
    ):
        """将提取的特征(ndarray)保存持久化保存(为npy文件)
        利用qtmd提供可视化特征抽取进度

        Parameters
        ----------
        partition : str
            "test|train"
        audio_paths_ : str
            音频文件的路径
        features_file_path : str
            保存文件名(路径)

        Returns
        -------
        ndarray
            提取的特征数组
        """
        features = self.extract_features(partition, audio_paths)
        # 保存数据
        if save_obj:
            np.save(features_file_path, features)

        return features

    def extract_features(self, partition=None, audio_paths=None):
        """
        Extract features from audio_paths for a specific partition.

        处理包括标准化放缩
        pca降维等特征优选操作

        Args:
        -
        - partition: str, the partition to extract features for (train, val, test).
        - audio_paths: List[str], the list of audio file paths to extract features from.
        - verbose: bool, whether or not to print debugging info.

        Returns:
        - 
        - features: np.ndarray, the extracted features as a numpy array.
        """
        features = self.extract_raw_features(partition=partition, audio_paths=audio_paths)

        # 考虑特征预处理
        from sklearn.preprocessing import StandardScaler

        # X为特征矩阵,y为标签

        fts = self.feature_transforms
        fts_keys=fts.keys()

        # if set(fts_keys) <= set(ava_fts_params):
        #     print("fts参数key合法")
        # else:
        #     print("fts参数不合法")
        for param in fts_keys:
            if param not in ava_fts_params:
                raise ValueError(f"fts参数{param}不合法,请参考可用的配置:{ava_fts_params}")
        print("fts参数key合法")

        if fts.get("std_scaler"):
            print("use StandardScaler to transform features")
            std_scaler = StandardScaler()
            features = std_scaler.fit_transform(features)
        # 小心字典关键字名字pca和pca_params,否则后面代码无法执行!
        pca_params_dict = fts.get("pca_params")
        
        if not pca_params_dict:
            pass
            # print("the pca params may be invalid!")
        print("🎈🎈🎈特征提取")
        if pca_params_dict:
            from sklearn.decomposition import PCA

            print("use PCA to transform features")

            n_components = pca_params_dict.get("n_components")

      
            if n_components=='mle':
                pass
            elif isinstance(n_components, int):
                pass
            elif n_components and n_components.isdigit():
                # if n_components.isdigit():
                # int()函数自带类型错误检测,有非法输入会自动抛出错误,所以这里直接使用,而不去手动检测输入的合法性
                # pca_params_dict['n_components'] = int(n_components)
                n_components=int(n_components)
            # elif n_components == "None":
            else:
                # pca_params_dict["n_components"] = None
                n_components=None

            # 将检验&处理后的n_components写入到pca字典中
            pca_params_dict['n_components']=n_components
            # 根据当前ae对象中的pca属性以及参数情况决定构造pca对象
            if self.pca is None:
                pca = self.pca = PCA(**pca_params_dict)
                pca.fit(features)

            else:
                pca = self.pca
            print(pca_params_dict, "@{pca_params_dict}😂")
            print(pca.n_components_, "@{pca.n_components_}")
            features = pca.transform(features)
            print(features.shape, "{features.shape}😂")
            # sys.exit()
            # 这部分可以抽取为单独的函数get_n_features_pca更加灵活
            # 使用面向对象的编程方式有点就显示出来了,可以通过对象属性或get方法来提高访问对象数据或属性,实现灵活通信,减少对于特定函数的依赖
            self.feature_dimension_pca = pca.n_components
        return features

    def get_dimensions(self):
        return np.array(self.test_features).shape[1]

    def extract_raw_features(self, partition=None, audio_paths=None):
        features = []
        # print(audio_paths)
        # append = features.append

        # 特征提取是一个比较耗时的过程,特征种类越多越耗时,这里采用tqdm显示特征提取进度条(已处理文件/文件总数)
        cnt = 0
        # iter = tqdm.tqdm(
        #     audio_paths, f"Extracting features for partition:{partition}"
        # )
        iter= audio_paths
        for audio_file in iter:

            if self.verbose > 1:
                cnt += 1
                if cnt % 20 == 0:
                    print(f"正在抽取第{cnt}个文件的特征..")
            # 调用utils模块中的extract_featrue进行特征提取
            f_config = self.f_config
            #! 抽取特征
            feature = extract_feature_of_audio(audio_file, f_config=f_config)
            if self.feature_dimension is None:
                # MCM特征组合下(3特征),有180维的单轴数组,5特征下,有193维
                self.feature_dimension = feature.shape[0]
            # 把当前文件提取出来特征添加到features数组中
            features.append(feature)

        # features是一个二维数组(n_samples,feature_dimension),每一行代表一个特征
        # 此时所有文件特征提取完毕,将其用numpy保存为npy文件
        # 构成二维数组
        features = np.array(features)
        return features

    def extract_update(self, partition="", meta_paths="", verbose=1):
        """
        根据meta_paths进行特征提取任务
        提取完特征后对相关self属性更新维护

        多次调用将执行增量提取,根据partition的取值,将每次的提取结果增量更新self的相应属性集

        Parameters
        ----------
        partition : str
            "train"|"test"
        meta_paths : list[str]|str
            需要提取特征的meta文件路径
        """
        if not meta_paths:
            raise ValueError("meta_files cannot be empty")
            # return meta_files
        meta_paths = self.pathlike_to_list(meta_paths)

        # 执行特征提取
        for meta_file in meta_paths:
            print(meta_file, "@🎈{meta_file}")
            # sys.exit()
            # 根据meta文件进行批量地特征提取
            features, audio_paths, emotions = self._extract_feature_in_meta(
                meta_path=meta_file, partition=""
            )
            # 这里将partition设置为空字符串,是因为标记特征是用来训练还是用来测试意义不大
            # 而且在跨库试验中,我们会让一个train/test features 交换身份,
            # 只需要知道这个特征文件包含哪些情感特征,来自哪个语料库,以及有多少个文件即可
            # 如果要更细致一些,可以考虑加入balance或shuffle信息,但这不是必须的,而且会对调试造成不变
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


    def load_data_preprocessing(self, meta_files=None, partition="", shuffle=False):
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
        print(f"{partition=}")
        # print(meta_files,"@{meta_files}in load_data_preprossing")
        if not meta_files:
            return
        self.extract_update(
            partition=partition,
            meta_paths=meta_files,
            # feature_transforms=self.feature_transforms,
        )

        # balancing the datasets ( both training or testing )
        if self.balance:
            self._balance_data(partition=partition)
        # shuffle
        if shuffle:
            self.shuffle_by_partition(partition)

    def shuffle_by_partition(self, partition):
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
                self.train_audio_paths,
                self.train_emotions,
                # self.train_features
                self.get_partition_features("train"),
            )
        elif partition == "test":
            (
                self.test_audio_paths,
                self.test_emotions,
                self.test_features,
            ) = shuffle_data(
                self.test_audio_paths,
                self.test_emotions,
                #   self.test_features
                self.get_partition_features("test"),
            )
        else:
            raise TypeError("Invalid partition, must be either train/test")

    def _balance_data(self, partition=""):
        """
        对训练集/测试集的数据做平衡处理

        """
        partition = validate_partition(partition=partition)

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

    def count_dict(self, verbose=0):
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
        partition = validate_partition(partition=partition, Noneable=False)
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
        # 手动在此处抛出调试性异常(此处采用pdb模块来调试)
        # raise ValueError("short!")
        audio_paths = [audio_paths[i] for i in p]
        emotions = [emotions[i] for i in p]
        features = [features[i] for i in p]

    return audio_paths, emotions, features


def load_data_from_meta(
    train_meta_files=None,
    test_meta_files=None,
    f_config=None,
    e_config=None,
    classification_task=True,
    shuffle=True,
    balance=False,
    feature_transforms=None,
) -> dict:
    """
    根据meta文件,提取/导入语音数据(numpy特征),并返回numpy打包train/test dataset相关属性的ndarray类型
    如果只想提取train/test dataset中的一方,那么另一方就传None(或者不传对应参数)即,前两个参数中允许其中一个为None

    Parameters
    ----------
    train_desc_files : list
        需要提取特征的语音文件列表信息,作为训练集
    test_desc_files : list
        需要提取特征的语音文件列表信息,作为测试集
    f_config : list[str], optional
        需要提取的特征组合, by default None
    e_config : list[str], optional
        需要使用的情感组合,类别字符串构成的列表, by default ['sad', 'neutral', 'happy']
    classification_task : bool, optional
        是否采用分类模型(否则使用回归模型), by default True
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
        shuffle=shuffle,
        verbose=True,
        feature_transforms_dict=feature_transforms,
    )

    # print(test_meta_files, "@{test_meta_files} in load_data_from_meta")

    # Loads training data
    ae.load_data_preprocessing(
        meta_files=train_meta_files, partition="train", shuffle=shuffle
    )
    # Loads testing data
    ae.load_data_preprocessing(
        meta_files=test_meta_files, partition="test", shuffle=shuffle
    )

    # 以train集为例检查self属性
    # print("ae.train_audio_paths:\n", ae.train_audio_paths)
    # X_train, X_test, y_train, y_test

    return {
        # "X_train": np.array(ae.train_features),
        # "X_test": np.array(ae.test_features),
        "X_train": ae.get_partition_features("train"),
        "X_test": ae.get_partition_features("test"),
        "y_train": np.array(ae.train_emotions),
        "y_test": np.array(ae.test_emotions),
        "train_audio_paths": np.array(ae.train_audio_paths),
        "test_audio_paths": np.array(ae.test_audio_paths),
        "balance": ae.balance,  # 反馈是否顺利执行了balance
        "ae": ae,
    }



def load_data_from_meta_demo():
    meta_dict=dict(
        train_meta_files=meta_dir/'train_emodb_HNS.csv',
        test_meta_files=meta_dir/'test_emodb_HNS.csv',
    )
    res=load_data_from_meta(**meta_dict,f_config=f_config_def)

    return res

if __name__ == "__main__":

    load_data_from_meta_demo()

    # ftd = dict(std_scaler=False, pca_params=dict(n_components=3))
    # ae = AudioExtractor(
    #     e_config=e_config_def,
    #     f_config=f_config_def, shuffle=True, feature_transforms_dict=ftd
    # )
    # print(ae)
    # ae._extract_feature_in_meta(meta_path=train_emodb_csv)
    # data = load_data_from_meta(
    #     # train_meta_files=train_emodb_csv,
    #     test_meta_files=test_emodb_csv,
    #     f_config=f_config_def,
    #     # balance=False,
    #     balance=True,
    # )
    
