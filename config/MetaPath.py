##
import collections
from pathlib import Path
from typing import List
from glob import glob

from sklearn.utils import deprecated

from config.EF import (
    AHNPS,
    ava_features,
    e_config_def,
    validate_emotions,
)

Sequence = collections.abc.Sequence  # type:ignore

# FileName = NewType('FileName', str)

# 控制保存文件的目录(例如相对于那个目录的相对路径或者直接采用绝对路径)
# project_dir = Path(r"D:/repos/CCSER/SER")

current_file=Path(__file__)
current_dir=current_file.parent
project_dir=(current_dir/"..").resolve()
# 尽管可以利用project_dir这个变量来指定保存文件的绝对路径,但是可以采用先cd到项目根目录,
# 然后执行`py .\tkinter_client\er_tk.py `的方式运行
# (而不是直接在./tkinter_client下执行`py er_tk.py`),这样相对路路径就会从项目根目录开始
meta_dir = Path("./meta_files")
grid_dir = Path("./grid")
features_dir = Path("./features")

emodb_files_glob = "data/emodb/wav/*.wav"
ravdess_files_glob = "data/ravdess/Actor_*"
savee_files_glob = "data/savee/AudioData/*/*.wav"


meta_dir, grid_dir, emodb_files_glob, ravdess_files_glob,savee_files_glob, features_dir = [
    (project_dir / p)
    for p in (meta_dir, grid_dir, emodb_files_glob, ravdess_files_glob,savee_files_glob, features_dir)
]
# 语料库配置
ravdess, emodb, savee = ["ravdess", "emodb", "savee"]
ava_dbs: list[str] = [emodb, ravdess, savee]

##
# !模型超参数路径
bclf1 = "bclf.joblib"
brgr1 = "brgr.joblib"
bclf2 = "bclf_u1.joblib"
brgr2 = "bclf_u1.joblib"
# bclf = bclf1
# brgr = brgr1
# 通过字典选取超参数版本(组合)
cuple_dict = dict(c1=(bclf1, brgr1), c2=(bclf2, brgr2))
bclf, brgr = cuple_dict["c1"]
# 补齐具体路径
bclf, brgr = [grid_dir / item for item in (bclf, brgr)]

def get_example_audio_file(db=savee):
    select_sample_dict=dict(
        emodb=emodb_files_glob,
        ravdess=ravdess_files_glob,
        savee=savee_files_glob
    )
    db_glob=str(select_sample_dict[db])
    g=glob(db_glob)
    return g[0]
##


def get_features_tag(f_config):
    """Returns label corresponding to which features are to be extracted
    返回形如('mfcc-chroma-contrast')的特征标签链

    params
    -
    f_config:list[str]|dict[str,bool]|str
        包含情感特征组合的可迭代对象

    Examples
    -
    eg1
    >>> f_config1 = {'mfcc': True, 'chroma': True, 'contrast': False, 'tonnetz': False, 'mel': False}
    >>> get_label(f_config1)
    >>> 'mfcc-chroma'
    >>> f_config2={'mfcc': True, 'chroma': True, 'contrast': True, 'tonnetz': False, 'mel': False}
    >>> utils.get_label(f_config2)
    >>> 'mfcc-chroma-contrast'

    eg2
    >>> MCM=['chroma', 'mel', 'mfcc']
    >>> get_features_tag(MCM)
    >>> 'chroma-mel-mfcc'
    """
    res = ""
    type_error = TypeError("Invalid type of f_config!")
    if f_config is None:
        return res
    if isinstance(f_config, dict):
        used_features = []
        for f in ava_features:
            if f_config.get(f):
                used_features.append(f)
        # used_features.sort()
        f_config = used_features
    elif isinstance(f_config, str):
        f_config = [f_config]
    # elif isinstance(f_config, Sequence):
    #     pass
    # else:
    #     raise type_error
    elif not isinstance(f_config, Sequence):
        print(type(f_config), type_error)
        raise type_error

    f_config.sort()
    res = "-".join(f_config)
    return res


##
def get_first_letters(emotions) -> str:
    """用于从一组情感标签列表中提取每个标签的首字母，并按字母顺序排序。具体来说，它接收一个字符串列表emotions，并返回一个字符串，该字符串包含按字母顺序排序的每个标签的首字母。

    params
    -
    emotions:list
        情感标签列表

    examples
    -
    以下是一个示例，演示如何使用该函数：
    emotions = ["happy", "sad", "angry", "excited"]
    print(get_first_letters(emotions))  # 输出：AEHS
    在这个例子中，我们定义了一个包含四个情感标签的列表emotions。我们将该列表传递给get_first_letters()函数，并打印函数返回的字符串表示形式。

    函数首先使用列表推导式从每个情感标签中提取首字母，并将其转换为大写字母。然后，它使用sorted()函数按字母顺序对所有首字母进行排序。最后，它使用"".join()函数将所有首字母连接为一个字符串，并返回该字符串。

    需要注意的是，如果emotions参数为空列表，则该函数将返回一个空字符串。此外，由于该函数只考虑每个标签的首字母，因此如果存在两个标签具有相同的首字母，则它们将在结果字符串中出现在一起。
    """
    res = ""
    if validate_emotions(emotions):
        res = "".join(sorted([e[0].upper() for e in emotions]))
    return res


def create_tag_name(
    db="",
    partition="",
    e_config=None,
    f_config=None,
    n_samples=0,
    ext="csv",
    balance=False,
    shuffle=False,
    **kwargs,
):
    """根据传入的参数,构造:
    1.meta_file文件名.csv
    2.numpy导出的ndarray文件.npy
    meta文件只是扫描语料库,和情感特征(features)没有关系

    Parameters
    ----------
    db : str
    partition : str
        "train"|"test"
    e_config : list
    f_config:list

    examples
    -
    >>> MCM=['chroma', 'mel', 'mfcc']
    >>> create_tag_name(emodb,f_config=MCM,n_samples=7,ext="npy")
    >>> 'emodb_chroma-mel-mfcc_7_npy'

    Returns
    -------
    str
        构造好的文件名
    """
    # if(partitionis None)
    features = get_features_tag(f_config)

    emotions = get_first_letters(e_config)

    # partition = tag_field(partition)
    # emotions = tag_field(emotions)
    # features = tag_field(features)
    # n_samples=tag_field(str(n_samples))
    # bool("")#False
    # bool(None)#False
    # res = f"{partition}{emotions}{features}{db}{n_samples}.{ext}"
    # str(None)#'None'
    n_samples = str(n_samples) if n_samples else ""
    # 如果是用来生成特征提取的文件名,可能需要加上额外的信息:
    balance = "balanced" if balance else ""
    shuffle = "shuffled" if shuffle else ""

    fields = [partition, db, features, emotions, n_samples]
    # print("{fields:}")
    # print(fields)
    # return
    true_fields = [f for f in fields if f]  # 标识非空的值
    # print(true_fields,"@{true_fields}")
    res = "_".join(true_fields) + f".{ext}"
    return res


def validate_partition(partition, Noneable=True):
    """判断partition是否合法字符串,否则原样返回

    Parameters
    ----------
    partition : str
        "train|test"时合法,否则非法,并抛出一个错误ValueError

    Returns
    -------
    str
        如果判断合法,返回原值(partition)

    Raises
    ------
    TypeError
        取值非法错误
    """
    if Noneable == False and partition == "":
        partition_invalid_raise_ValueError(partition=partition)
    if partition:
        res = partition in ["test", "train"]
        if not res:
            partition_invalid_raise_ValueError(partition=partition)
    return partition


def partition_invalid_raise_ValueError(partition=""):
    partition = partition if partition else ""
    raise TypeError(
        f"Unknown partition @{partition} only 'train' or 'test' is accepted"
    )


##
@deprecated()
def tag_field(field):
    """为非空字段转换为字符串,并添加下划线

    Parameters
    ----------
    field : str|int
        字段

    Returns
    -------
    str
        处理好的字符串
    """
    if field:
        field += "_"
    else:
        field = ""
    return field


def prepend_dir(tag_names, dir=meta_dir, change_type="Path"):
    """由于存放meta文件的目录不在项目根目录(而是子目录meta_files,因此这里统一做一个转换)
    如果需要使用绝对路径,可以在此处修改(本函数管理一切目录前缀)
    Parameters
    ----------
    tag_names : list[str]|str
        就是文件名(不带任何目录前缀),本函数将为其加上前缀
    type:str|bool
        是否将处理好的路径转换为str类型


    Returns
    -------
    List[type]|type
    即(List[Path]|List[str])|(Path|str)
        从项目根目录开始的具体路径
    """
    # 将输入统一为列表
    if isinstance(tag_names, str):
        tag_names = [tag_names]
    elif not isinstance(tag_names, Sequence):
        raise TypeError(f"tag_names@{tag_names} must be a Sequence!")
    # 返回类型转换(使用绝对路径不容易出问题,但是相对路径性能更好)
    res = [(dir / file).absolute() for file in tag_names]
    if change_type == "str":
        res = [str(meta_path) for meta_path in res]
    elif change_type != "Path":
        raise ValueError("type must be path_like:`Path` or 'str' of path")
    # 如果tag_names只有一个元素,那么返回单个元素
    # res=res[0] if len(res)==1 else res
    return res


# def meta_names(db,e_config=e_config):


def meta_names_of_db(db, e_config=None):
    """根据指定的预料数据库分别构造训练姐和测试集meta文件名

    Parameters
    ----------
    db : str
        指定的语料库
    e_config : list, optional
        情感组合, by default e_config

    Returns
    -------
    dict|tuple[str,str]
        构造好的字典|构造好的名字元组
        字典同样可以拆包(成组赋值)
    """
    train = create_tag_name(db, partition="train", e_config=e_config)
    test = create_tag_name(db, partition="test", e_config=e_config)
    return train, test

    # return {
    #     "train": train,
    #     "test": test
    # }


##
def meta_paths_of_db(db, e_config=None, partition="", change_type="Path"):
    """根据指定的语料库和情感特征组合,生成具体的train/test set meta file (csv)路径

    Parameters
    ----------
    db : str
        语料库的名字
    e_config : list[str], optional
        情感特征配置(组合), by default e_config_def

    Returns
    -------
    tuple
        train/test set meta file (csv)

    examples
    -

    >>> meta_paths_of_db(emodb,e_config=AHNPS)
    >>>
        [WindowsPath('meta_files/train_emodb_AHNPS.csv'),
        WindowsPath('meta_files/test_emodb_AHNPS.csv')]
    """
    names = meta_names_of_db(db, e_config=e_config)
    # print(names,"@{names}")
    res = prepend_dir(names, change_type=change_type)
    # print(res,"@{res}")
    # 是否根据paritition参数仅返回train/test中的一个meta文件
    partition = validate_partition(partition)

    if partition:
        train, test = res
        if partition == "train":
            res = train
        else:
            res = test
    return res


def _meta_names_all(dbs=ava_dbs, partition="", e_config=None):
    """根据数据库列表dbs中,构造训练集和测试集meta文件名

    Parameters
    ----------
    partition : str
        "test"|"train"
    e_config : list, optional
        情感组合, by default e_config

    Returns
    -------
    list
        meta文件名list
    """
    if isinstance(dbs, str):
        dbs = [dbs]
    meta_partition = [create_tag_name(db, partition, e_config=e_config) for db in dbs]
    return meta_partition


def create_meta_paths_dbs(dbs=ava_dbs, partition="", e_config=None):
    """计算dbs中配置的所有语料库的train/test meta文件路径

    Parameters
    ----------
    dbs=dbs: list|str
        语料库列表, by default dbs
        通常只需要选择一个语料库即可
    e_config : list, optional
        情感组合, by default e_config_def

    Returns
    -------
    list
        路径列表

    e.g.
    -
    >>> meta_paths_dbs(dbs=[emodb],partition="test")
    >>> [WindowsPath('meta_files/test_emodb_HNS.csv')]

    >>> meta_paths_dbs(dbs=[ravdess,emodb],partition="test")
    >>>
         [WindowsPath('meta_files/test_ravdess_HNS.csv'),
         WindowsPath('meta_files/test_emodb_HNS.csv')]

    >>> meta_paths_bs(dbs=[emodb,ravdess])
    >>>
    [WindowsPath('meta_files/train_emodb_HNS.csv'),
     WindowsPath('meta_files/test_emodb_HNS.csv'),
     WindowsPath('meta_files/train_ravdess_HNS.csv'),
     WindowsPath('meta_files/test_ravdess_HNS.csv')]
    """
    meta_trains = _meta_names_all(dbs=dbs, partition="train", e_config=e_config)
    meta_tests = _meta_names_all(dbs=dbs, partition="test", e_config=e_config)

    res = []
    train_paths = prepend_dir(meta_trains)
    test_paths = prepend_dir(meta_tests)

    partition = validate_partition(partition)
    if partition:
        if partition == "train":
            res = train_paths
        else:
            res = test_paths
    else:
        pairs = zip(train_paths, test_paths)
        for pair in list(pairs):
            res += list(pair)

    return res


def test1():
    res = create_tag_name(db="emodb", e_config=e_config_def, f_config=None)
    print(res)
    res1 = create_tag_name(db="ravdess", partition="test")
    print(res1)


def test2():
    res = create_meta_paths_dbs(e_config=None)
    for path in res:
        print(path)


##

partition_meta_files = create_meta_paths_dbs(e_config=e_config_def)

(
    train_emodb_csv,
    test_emodb_csv,
    train_ravdess_csv,
    test_ravdess_csv,
    train_savee_csv,
    test_savee_csv,
) = [str(meta) for meta in partition_meta_files]

pair1 = (train_ravdess_csv, test_ravdess_csv)
pair2 = (train_emodb_csv, test_emodb_csv)
pair3 = (train_ravdess_csv, test_emodb_csv)
pair4 = (train_emodb_csv, test_ravdess_csv)
pair5 = (train_emodb_csv, train_ravdess_csv)

paris = [pair1, pair2, pair3, pair4, pair5]


def meta_pairs(e_config=None):
    """获取语料库分组组合
    注意,meta文件没有包含特征信息,因此只需要指定e_config即可

    Parameters
    ----------
    e_config : list[str], optional
        需要试验的情感范围, by default e_config_def

    Returns
    -------
    list[tuple]
        返回若干组合
    """
    partition_meta_files = create_meta_paths_dbs(e_config=e_config)

    train_emodb_csv, test_emodb_csv, train_ravdess_csv, test_ravdess_csv = [
        str(meta) for meta in partition_meta_files
    ]

    pair1 = (train_ravdess_csv, test_ravdess_csv)
    pair2 = (train_emodb_csv, test_emodb_csv)
    pair3 = (train_ravdess_csv, test_emodb_csv)
    pair4 = (train_emodb_csv, test_ravdess_csv)
    pair5 = (train_emodb_csv, train_ravdess_csv)

    pairs = [pair1, pair2, pair3, pair4, pair5]
    return pairs


def select_meta_dict(pair=pair1):
    print(f"当前使用语料库数据集组合:\n{pair}")
    meta_dict = {"train_meta_files": pair[0], "test_meta_files": pair[1]}
    return meta_dict


##


##
if __name__ == "__main__":
    # print(train_emodb_csv, train_emodb_csv.absolute())
    # test1()
    # test2()
    # res = meta_paths(ravdess)
    # print(partition_meta_files)
    # create_meta_paths_dbs()
    meta_paths_of_db(db=emodb, e_config=AHNPS)


##
