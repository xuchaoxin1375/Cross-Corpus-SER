##
import os
import random
from glob import glob
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split as tts

from config.EF import categories_emodb, extend_emotion_names
from config.MetaPath import (
    emodb,
    emodb_files_glob,
    meta_paths_of_db,
    ravdess,
    ravdess_files_glob,
    savee,
    savee_files_glob,
)


def check_meta_names(e_config, train_name=None, test_name=None, db=""):
    if train_name is None or test_name is None:
        train_name, test_name = meta_paths_of_db(db, e_config=e_config)
    if db == "":
        raise ValueError("db must be specified non None")
    return train_name, test_name


##
"""
本模块负责将不同的语料库文件的元数据(路径和情感标签)统一化,并且作持久化保存处理(csv文件)
不同的语料库由于文件名规范不同,所以需要分别编写处理函数
!注意小心使用os.chdir()改变工作目录,特别需要访问不同目录的情况
"""


def create_emodb_csv(
    e_config=None,
    train_name=None,
    test_name=None,
    train_size=0.8,
    verbose=1,
    shuffle=False,
):
    """
    Reads speech emodb dataset from directory and write it to a metadata CSV file.

    params:
    -
        emotions (list):
            list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str):
            the output csv filename for training data, default is 'train_emo.csv'
        test_name (str):
            the output csv filename for testing data, default is 'test_emo.csv'
        train_size (float):
            the ratio of splitting training data, default is 0.8 (80% Training data and 20% testing data)

            - 可以考虑使用sklearn的model_section模块提供的api来完成数据划分操作,例如train_test_split()方法
        verbose (int/bool):
             verbositiy level, 0 for silence, 1 for info, default is 1
    """
    db = emodb

    train_name, test_name = check_meta_names(e_config, train_name, test_name, db)

    target = {"path": [], "emotion": []}
    categories = categories_emodb
    tp = target["path"]
    te = target["emotion"]
    # delete not specified emotions
    # 颠倒字典种的键值对为 情感:缩写(emotion:code)
    # 颠倒不是必须的,这里顺便为之,categories_reversed作为临时的字典辅助筛掉不需要的情感
    categories_reversed = {emotion: code for code, emotion in categories.items()}
    if e_config is None:
        raise ValueError(f"e_config is None")
    for emotion, code in categories_reversed.items():
        if emotion not in e_config:
            del categories[code]
    # shuffle可以在这个环节进行:
    files_iter = glob(emodb_files_glob)
    if shuffle:
        random.shuffle(files_iter)

    for file in files_iter:
        try:
            # emodb文件形如`03a01Wa.wav`
            # - 03 表示这个音频记录来自第3位演员
            # - a01 表示这个音频记录是该演员表演的第1种情感
            # - W 表示这个情感是“愤怒”（Angry）的缩写
            # - a 表示这个是该情感的第1个副本（第一个表演）
            # - .wav 表示这个文件的格式为.wav格式
            # 我们感兴趣的是第6个字符(也即是[5])
            e = os.path.basename(file)[5]
            emotion: str = categories[e]
            # 可以使用字典的get方法来避免KeyError,不过这里我们考虑通过异常抛出做出及时的反馈
        except KeyError:
            # print("key error")
            continue
        tp.append(file)
        te.append(emotion)
        # target['emotion'].append(emotion)
        # target['path'].append(file)
    n_samples = len(target["path"])

    if verbose:
        print("[EMO-DB] Total files to write:", n_samples)

    # dividing training/testing sets
    test_samples: int = int((1 - train_size) * n_samples)
    train_samples: int = int(train_size * n_samples)
    if verbose:
        print(
            f"[{db}_{e_config}] Training samples:",
            train_samples,
            "\nTesting samples:",
            test_samples,
        )

    X_train = target["path"][:train_samples]
    X_test = target["path"][train_samples:]
    y_train = target["emotion"][:train_samples]
    y_test = target["emotion"][train_samples:]
    print(e_config)

    # train_name=create_tag_name(emodb,partition="train")
    # test_name=create_tag_name(emodb,partition="test")
    # print(X_train,y_train)
    # print(train_name,test_name)
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)


##
def create_savee_csv(
    e_config=None,
    train_name=None,
    test_name=None,
    train_size=0.8,
    verbose=1,
    subset_size=1,
    shuffle=False,
):
    db = savee
    train_name, test_name = check_meta_names(e_config, train_name, test_name, db)
    print(train_name, "@{train_name}")
    print(test_name, "@{test_name}")
    import re

    # 数据量不能太小,否则tts分割可能会因为train/test set中的某个会是空的,导致报错
    audios = glob(savee_files_glob)
    total = len(audios)
    audios = audios[: int(total * subset_size)]

    from config.EF import categories_savee

    emos = []
    paths = []
    for audio in audios:
        audio_name = os.path.basename(audio)
        name, ext_ = os.path.splitext(audio_name)
        # 定义一个正则表达式，用于匹配开头的单词
        pattern = r"[a-zA-Z]+"

        # 使用 re 模块的 findall 函数查找所有匹配项
        m = re.match(pattern, name)
        # print(audio_name,m)
        # 打印出所有匹配项的值
        emo_code = m.group()
        # print(emo_code)
        emo = categories_savee[emo_code]
        # print(audio,emo)
        paths.append(audio)
        emos.append(emo)

    df = DataFrame({"path": paths, "emotion": emos})
    # emo_bool_mask=df["emotion" ]
    emo_bool_mask = df["emotion"].isin(e_config)
    df = df[emo_bool_mask]
    print(f"{df.shape=}")

    n_samples = len(df)
    if verbose:
        print("[savee] Total files to write:", n_samples)
    # dividing training/testing sets
    test_samples: int = int((1 - train_size) * n_samples)
    train_samples: int = int(train_size * n_samples)
    if verbose:
        print(
            f"[{db}_{e_config}] Training samples:",
            train_samples,
            "\nTesting samples:",
            test_samples,
        )
    # paths=df["path"].tolist()
    # emos=df["emotion"].tolist()

    e_config_paths = df["path"]
    e_config_emos = df["emotion"]
    spl = tts(e_config_paths, e_config_emos, train_size=train_size, shuffle=shuffle)
    X_train, X_test, y_train, y_test = spl
    # df_e_config=

    print(train_name, "@{train_name}")
    print(test_name, "@{test_name}")
    DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name, index=False)
    DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name, index=False)
    print("文件创建完毕!")
    return spl


# test
# res=create_savee_csv(e_config=e_config_def)


##


target_debug = []


def create_ravdess_csv(
    e_config=None, train_name=None, test_name=None, train_size=0.75, verbose=1
):
    """
    Reads speech training(RAVDESS) datasets from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is e_config
        train_name (str): the output csv filename for training data, default is 'train_RAVDESS.csv'
        test_name (str): the output csv filename for testing data, default is 'test_RAVDESS.csv'
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    db = ravdess
    target = {"path": [], "emotion": []}
    # 声明外部变量,用于临时调试!
    global target_debug
    target_debug = target

    # 这个数据库文件比较多,为了敏捷性,控制只处理特定情感的文件而不是全部情感文件
    # 我们将TESS,RAVDESS语料库放在了训练集目录training
    print("[RAVDESS] files meta extacting...")
    if e_config is None:
        raise ValueError("[RAVDESS] e_config is None")
    for e in e_config:
        # for training speech directory
        ravdess_file_glob_emo = f"{ravdess_files_glob}/*_{e}.wav"
        total_files = glob(ravdess_file_glob_emo)
        for i, path in enumerate(total_files):
            target["path"].append(path)
            target["emotion"].append(e)
        # 提示所有训练集文件meta处理完毕
        if verbose and total_files:
            print(f"There are {len(total_files)} training audio files for category:{e}")
    target = DataFrame(target)
    # print(target)
    # train_name,test_name =file_names(train_name,test_name)
    train_name, test_name = meta_paths_of_db(db, e_config=e_config)
    X_train, X_test = tts(target, test_size=1-train_size, random_state=0)
    print(X_train[:5], "@{X_train}")
    print(train_name, "@{train_name}")
    DataFrame(X_train).to_csv(train_name)
    DataFrame(X_test).to_csv(test_name)
    if verbose:
        print("ravdess db was splited to 2 csv files!")
        print(f"the train/test size rate is:{train_size}:{(1-train_size)}")


# 不可以挪到顶部,因为下面的selector的定义需要用到上面定义的函数
selector = {
    emodb: create_emodb_csv,
    ravdess: create_ravdess_csv,
    savee: create_savee_csv,
}


def create_csv_by_metaname(meta_file):
    """根据给定的符合本项目的文件名构造规范的文件名,生成对应的train/test dataset metadata files

    Parameters
    ----------
    meta_file : str
        文件名
    """
    print(meta_file, "@{meta_file}to be create...😂")
    name = Path(meta_file).name
    # print(name,"@{name}")
    name, ext = os.path.splitext(name)  # 文件名去掉后缀ext
    # 直接解析成三个字符串
    # print(name,"@{name}")
    _partitoin, db, emotion_first_letters = name.split("_")

    e_config = extend_emotion_names(emotion_first_letters)

    field_p2 = [db, emotion_first_letters]
    train_name = "_".join(["train"] + field_p2) + ".csv"
    test_name = "_".join(["test"] + field_p2) + ".csv"
    print(f"@create_csv..🎈{train_name}\n{test_name}")

    selector[db](
        e_config=e_config,
        #   train_name=train_name,
        #   test_name=test_name
    )


def create_meta_csv(
    train_meta_files,
    test_meta_files,
    dbs=None,
    e_config=None,
    verbose=1,
    override_csv=False,
):
    """
    @deprecated
    Write available CSV files in `self.train_desc_files` and `self.test_desc_files`
    determined by `self._set_metadata_filenames()` method.

    ## Note:
    硬编码实现:
    if emodb in train_csv_file:
            write_emodb_csv(
                self.e_config,
                train_name=train_csv_file,
                test_name=test_csv_file,
                verbose=self.verbose,
            )
            if self.verbose:
                print("[I] Generated EMO-DB  CSV meta File")
        elif ravdess in train_csv_file:
            write_ravdess_csv(
                self.e_config,
                train_name=train_csv_file,
                test_name=test_csv_file,
                verbose=self.verbose,
            )
            if self.verbose:
                print("[I] Generated RAVDESS CSV meta File")
    """
    meta_handler_dict = {emodb: create_emodb_csv, ravdess: create_ravdess_csv}
    for train_csv_file, test_csv_file in zip(train_meta_files, test_meta_files):
        # 使用Path对象的`/`操作符连接路径
        # train_csv_file = (meta_dir / train_csv_file).name
        # test_csv_file = (meta_dir / test_csv_file).name
        # 兼容性的写法
        if os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
            # file already exists, just skip writing csv files
            if not override_csv:
                continue
        if dbs:
            for db in dbs:
                if meta_handler_dict.get(db) is None:
                    raise ValueError(f"{db} not recognized")
                meta_handler_dict[db](
                    e_config,
                    train_name=train_csv_file,
                    test_name=test_csv_file,
                    verbose=verbose,
                )
                if verbose:
                    print(f"[I] Generated {db} CSV meta File")


##
if __name__ == "__main__":
    # write_emodb_csv(e_config=AHNPS)
    # write_ravdess_csv()
    name1 = "test_emodb_AS.csv"
    # create_csv_by_metaname(name1)
    name2 = "train_savee_AS.csv"
    create_csv_by_metaname(name2)

    ##
    # import numpy as np
    # from sklearn.model_selection import train_test_split

    # M=np.arange(100).reshape(10,10)
    # print(M,"@{M}")
    # X_train,X_test=train_test_split(M,train_size=0.7,shuffle=False)
    # print(X_train,"@{X_train}")
    # print(X_test,"@{X_test}")
