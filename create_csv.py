##
import glob
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split as tts

import MetaPath
from EF import AHNPS, categories_emodb, extend_names
from MetaPath import (create_tag_name, emodb, emodb_files_glob, meta_dir,
                      meta_names_of_db, ravdess, test_emodb_csv,
                      test_ravdess_csv, train_emodb_csv, train_ravdess_csv,meta_paths_of_db)

##
"""
本模块负责将不同的语料库文件的元数据(路径和情感标签)统一化,并且作持久化保存处理(csv文件)
不同的语料库由于文件名规范不同,所以需要分别编写处理函数
!注意小心使用os.chdir()改变工作目录,特别需要访问不同目录的情况
"""



from EF import e_config_def


def write_emodb_csv(e_config=e_config_def, train_name=None,
                    test_name=None, train_size=0.8, verbose=1,shuffle=False):
    """
    Reads speech emodb dataset from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_emo.csv'
        test_name (str): the output csv filename for testing data, default is 'test_emo.csv'
        train_size (float): the ratio of splitting training data, default is 0.8 (80% Training data and 20% testing data)
        # 可以考虑使用sklearn的model_section模块提供的api来完成数据划分操作,例如train_test_split()方法
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    db=emodb
    target = {"path": [], "emotion": []}
    categories=categories_emodb
    tp=target["path"]
    te=target["emotion"]
    # delete not specified emotions
    # 颠倒字典种的键值对为 情感:缩写(emotion:code)
    #颠倒不是必须的,这里顺便为之,categories_reversed作为临时的字典辅助筛掉不需要的情感
    categories_reversed = { emotion: code for code, emotion in categories.items() }
    for emotion, code in categories_reversed.items():
        if emotion not in e_config:
            del categories[code]
    # shuffle可以在这个环节进行:
    files_iter=glob.glob(emodb_files_glob)
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
            e=os.path.basename(file)[5]
            emotion:str = categories[e]
            # 可以使用字典的get方法来避免KeyError,不过这里我们考虑通过异常抛出做出及时的反馈
        except KeyError:
            # print("key error")
            continue
        tp.append(file)
        te.append(emotion)
        # target['emotion'].append(emotion)
        # target['path'].append(file)
    n_samples=len(target['path'])

    if verbose:
        print("[EMO-DB] Total files to write:", n_samples)
        
    # dividing training/testing sets
    test_samples:int = int((1-train_size) * n_samples)
    train_samples:int= int(train_size * n_samples)
    if verbose:
        print(f"[{db}_{e_config}] Training samples:", train_samples,"\nTesting samples:", test_samples)   
    
    X_train = target['path'][:train_samples]
    X_test = target['path'][train_samples:]
    y_train = target['emotion'][:train_samples]
    y_test = target['emotion'][train_samples:]
    print(e_config)
    train_name,test_name = meta_paths_of_db(emodb,e_config=e_config)
    # print(train_name,test_name)
    # train_name=create_tag_name(emodb,partition="train")
    # test_name=create_tag_name(emodb,partition="test")
    # print(X_train,y_train)
    # print(train_name,test_name)
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)

target_debug=[]
def write_ravdess_csv(emotions=e_config_def, train_name=None,
                            test_name=None,train_size=0.75, verbose=1):
    """
    Reads speech training(RAVDESS) datasets from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_RAVDESS.csv'
        test_name (str): the output csv filename for testing data, default is 'test_RAVDESS.csv'
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    db=ravdess
    target = {"path": [], "emotion": []}
    # 声明外部变量,用于临时调试!
    global target_debug
    target_debug=target
    
    # 这个数据库文件比较多,为了敏捷性,控制只处理特定情感的文件而不是全部情感文件
    # 我们将TESS,RAVDESS语料库放在了训练集目录training
    print("[RAVDESS] files meta extacting...")
    for e in emotions:
        # for training speech directory
        total_files = glob.glob(f"data/ravdess/Actor_*/*_{e}.wav")
        for i, path in enumerate(total_files):
            target["path"].append(path)
            target["emotion"].append(e)
        # 提示所有训练集文件meta处理完毕
        if verbose and total_files:
            print(f"There are {len(total_files)} training audio files for category:{e}")
    target=DataFrame(target)
    # print(target)
    # train_name,test_name =file_names(train_name,test_name)
    train_name,test_name=meta_paths_of_db(db,e_config=e_config_def)
    X_train,X_test = tts(target,train_size=train_size,random_state=0)
    # print(X_train,X_test)
    DataFrame(X_train).to_csv(train_name)
    DataFrame(X_test).to_csv(test_name)
    if verbose: 
        print("ravdess db was splited to 2 csv files!")
        print(f"the train/test size rate is:{train_size}:{(1-train_size)}")
def write_ravdess_csv_bak(emotions=e_config_def, train_name=train_ravdess_csv,
                            test_name=test_ravdess_csv,train_size=0.75, verbose=1):
    """
    Reads speech training(RAVDESS) datasets from directory and write it to a metadata CSV file.
    params:
        emotions (list): list of emotions to read from the folder, default is ['sad', 'neutral', 'happy']
        train_name (str): the output csv filename for training data, default is 'train_RAVDESS.csv'
        test_name (str): the output csv filename for testing data, default is 'test_RAVDESS.csv'
        verbose (int/bool): verbositiy level, 0 for silence, 1 for info, default is 1
    """
    train_target = {"path": [], "emotion": []}
    test_target = {"path": [], "emotion": []}
    
    # 这个数据库文件比较多,为了敏捷性,控制只处理特定情感的文件而不是全部情感文件
    # 我们将TESS,RAVDESS语料库放在了训练集目录training
    for e in emotions:
        # for training speech directory
        total_files = glob.glob(f"data/training/Actor_*/*_{e}.wav")
        for i, path in enumerate(total_files):
            train_target["path"].append(path)
            train_target["emotion"].append(e)
        # 提示所有训练集文件meta处理完毕
        if verbose and total_files:
            print(f"[RAVDESS] There are {len(total_files)} training audio files for category:{e}")

        #将验证集(测试集)文件放到data/validation目录
        # for validation speech directory
        total_files = glob.glob(f"data/validation/Actor_*/*_{e}.wav")
        for i, path in enumerate(total_files):
            test_target["path"].append(path)
            test_target["emotion"].append(e)
        # 提示所有验证集文件meta处理完毕
        if verbose and total_files:
            print(f"[RAVDESS] There are {len(total_files)} testing audio files for category:{e}")
    
    pd.DataFrame(test_target).to_csv(test_name)
    pd.DataFrame(train_target).to_csv(train_name)

def create_csv_by_metaname(meta_file):
    """根据给定的符合本项目的文件名构造规范的文件名,生成对应的train/test dataset metadata files

    Parameters
    ----------
    meta_file : str
        文件名
    """
    from pathlib import Path
    name=Path(meta_file).name
    name,_=os.path.splitext(name)#文件名去掉后缀ext
    # 直接解析成三个字符串
    _partitoin,db,emotion_first_letters=name.split("_")
    
    selecter={emodb:write_emodb_csv,ravdess:write_ravdess_csv}
    e_config=extend_names(emotion_first_letters)
    
    field_p2=db+emotion_first_letters
    train_name="_".join(["train",field_p2])
    test_name="_".join(["test",field_p2])
    
    selecter[db](e_config=e_config,train_name=train_name,test_name=test_name)

##
if __name__=="__main__":
    # write_emodb_csv(e_config=AHNPS)
    # write_ravdess_csv()
    name='test_emodb_AHS.csv'
    create_csv_by_metaname(name)