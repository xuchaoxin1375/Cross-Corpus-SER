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

emodb_files_glob, ravdess_files_glob, savee_files_glob = [
    str(p) for p in [emodb_files_glob, ravdess_files_glob, savee_files_glob]
]


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


def create_emodb_meta(
    e_config=None,
    train_name=None,
    test_name=None,
    train_size=0.8,
    verbose=1,
    shuffle=True,
    balance=False,
    sort=True,
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
            the ratio of splitting training data, default is 0.8

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
    # print(e_config)

    # train_name=create_tag_name(emodb,partition="train")
    # test_name=create_tag_name(emodb,partition="test")
    # print(X_train,y_train)
    # print(train_name,test_name)
    write_to_csv(train_name, test_name, sort, X_train, X_test, y_train, y_test)


def write_to_csv(train_name, test_name, sort, X_train, X_test, y_train, y_test):
    df_train = DataFrame({"path": X_train, "emotion": y_train})
    df_test = DataFrame({"path": X_test, "emotion": y_test})
    if sort:
        df_train.sort_values(by="emotion").to_csv(train_name)
        df_test.sort_values(by="emotion").to_csv(test_name)


##
def create_savee_meta(
    e_config=None,
    train_name=None,
    test_name=None,
    train_size=0.8,
    verbose=1,
    subset_size=1,
    shuffle=True,
    balance=False,
    sort=True,
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

    p_e_df = DataFrame({"path": paths, "emotion": emos})
    # emo_bool_mask=df["emotion"]
    emo_bool_mask = p_e_df["emotion"].isin(e_config)
    p_e_df = p_e_df[emo_bool_mask]
    if verbose:
        print(f"{p_e_df.shape=}")
        n_samples = len(p_e_df)
        print("[savee] Total files to write:", n_samples)
        # dividing training/testing sets
        test_samples: int = int((1 - train_size) * n_samples)
        train_samples: int = int(train_size * n_samples)
        print(
            f"[{db}_{e_config}] Training samples:",
            train_samples,
            "\nTesting samples:",
            test_samples,
        )

    # paths = p_e_df["path"]
    # emotions = p_e_df["emotion"]
    # spl = tts(paths, emotions, train_size=train_size, shuffle=shuffle)
    # X_train, X_test, y_train, y_test = spl

    # DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name, index=False)
    # DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name, index=False)
    # write_to_csv(train_name, test_name, sort, X_train, X_test, y_train, y_test)

    spl = tts(p_e_df, train_size=train_size, shuffle=shuffle)
    Xy_train, Xy_test = spl
    from_df_write_to_csv(train_name=train_name, test_name=test_name, sort=sort, Xy_train=Xy_train, Xy_test=Xy_test)

    if verbose:
        print(train_name, "@{train_name}")
        print(test_name, "@{test_name}")
        print("文件创建完毕!")
    return spl


##


def create_ravdess_meta(
    e_config=None,
    train_name=None,
    test_name=None,
    train_size=0.75,
    verbose=1,
    shuffle=True,
    balance=False,
    sort=True,
):
    """
    Reads speech training(RAVDESS) datasets from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is e_config
        train_name (str): the output csv filename for training data,
        test_name (str): the output csv filename for testing data,
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    db = ravdess
    meta_dict = {"path": [], "emotion": []}

    # 这个数据库文件比较多,为了敏捷性,控制只处理特定情感的文件而不是全部情感文件
    # 我们将TESS,RAVDESS语料库放在了训练集目录training
    print(f"{db} files meta extacting...")
    if e_config is None:
        raise ValueError(f"{db}e_config is None")
    for e in e_config:
        # for training speech directory
        ravdess_file_glob_emo = f"{ravdess_files_glob}/*_{e}.wav"
        total_files = glob(ravdess_file_glob_emo)
        for i, path in enumerate(total_files):
            meta_dict["path"].append(path)
            meta_dict["emotion"].append(e)
        # 提示所有训练集文件meta处理完毕
        if verbose and total_files:
            print(f"There are {len(total_files)} training audio files for category:{e}")
    meta_df = DataFrame(meta_dict)
    # print(target)
    train_name, test_name = meta_paths_of_db(db, e_config=e_config)
    Xy_train, Xy_test = tts(
        meta_df, train_size=train_size, random_state=0, shuffle=shuffle
    )
    print(Xy_train[:5], "@{X_train}")
    print(train_name, "@{train_name}")
    from_df_write_to_csv(train_name=train_name, test_name=test_name, sort=sort, Xy_train=Xy_train, Xy_test=Xy_test)

    if verbose:
        print("ravdess db was splited to 2 csv files!")
        print(f"the train/test size rate is:{train_size}:{(1-train_size)}")


def from_df_write_to_csv(train_name="", test_name="", sort=True, Xy_train=None, Xy_test=None):
    train_df = DataFrame(Xy_train)
    test_df = DataFrame(Xy_test)
    if sort:
        sorted_train_df = train_df.sort_values(by="emotion")
        sorted_test_df = test_df.sort_values(by="emotion")
    sorted_train_df.to_csv(train_name)
    sorted_test_df.to_csv(test_name)


# 不可以挪到顶部,因为下面的selector的定义需要用到上面定义的函数
selector = {
    emodb: create_emodb_meta,
    ravdess: create_ravdess_meta,
    savee: create_savee_meta,
}


def create_csv_by_metaname(meta_file, shuffle=True):
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
        shuffle=shuffle,
        #   train_name=train_name,
        #   test_name=test_name
    )


##
if __name__ == "__main__":
    # write_emodb_csv(e_config=AHNPS)
    # write_ravdess_csv()
    name1 = "test_emodb_AS.csv"
    # create_csv_by_metaname(name1)
    name2 = "train_savee_AS.csv"
    name3 = "test_savee_HNS.csv"
    create_csv_by_metaname(name3, shuffle=True)

    ##
    # import numpy as np
    # from sklearn.model_selection import train_test_split

    # M=np.arange(100).reshape(10,10)
    # print(M,"@{M}")
    # X_train,X_test=train_test_split(M,train_size=0.7,shuffle=False)
    # print(X_train,"@{X_train}")
    # print(X_test,"@{X_test}")
