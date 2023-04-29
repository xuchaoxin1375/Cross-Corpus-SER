##

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile
from joblib import load
from sklearn.preprocessing import StandardScaler

from audio.converter import convert_audio
from config.EF import MCM, ava_features, f_config_def
from config.MetaPath import bclf, brgr, project_dir,speech_dbs_dir


def get_used_keys(config_dict):
    """将传入的字典中值为True的key添加到列表中并返回
    
    examples
    -
    >>> get_used_keys({"a":123,"b":False,"c":"nice"})
    >>> ['a','c']
    
    """    
    used_keys = []
    for key, value in config_dict.items():
        if value:
            used_keys.append(key)
    return used_keys

def get_dropout_str(dropout, n_layers=3):
    """这是一个Python函数，用于将给定的Dropout参数转换为字符串。

    params:
    -
    dropout:list[float]|float
        如果dropout是一个列表，则函数将返回用下划线分隔的所有元素的字符串表示形式。
        如果dropout是一个浮点数，则函数将返回长度为n_layers的下划线分隔字符串，
        其中每个元素都是dropout的字符串表示形式。

    n_layers:int,default is 3
        指定层数

    examples:
    -
    以下是一个示例，演示如何使用该函数：

    dropout1 = 0.2
    dropout2 = [0.2, 0.5, 0.7]
    print(get_dropout_str(dropout1))  # 输出：0.2_0.2_0.2
    print(get_dropout_str(dropout2))  # 输出：0.2_0.5_0.7
    在这个例子中，我们定义了两个不同的Dropout参数dropout1和dropout2。
    我们分别将它们传递给get_dropout_str()函数，并打印函数返回的字符串表示形式。

    对于dropout1，函数将返回一个长度为3的下划线分隔字符串，其中每个元素都是0.2的字符串表示形式。
    对于dropout2，函数将返回一个包含所有元素的下划线分隔字符串的字符串表示形式。
    请注意，如果dropout参数既不是列表也不是浮点数，则该函数可能会引发异常。
    此外，如果n_layers参数大于传递的dropout参数的长度，则函数将使用最后一个元素来填充字符串。"""
    if isinstance(dropout, list):
        return "_".join([str(d) for d in dropout])
    elif isinstance(dropout, float):
        return "_".join([str(dropout) for _ in range(n_layers)])


def extract_feature_of_audio(audio_file_name, f_config):
    """
    用于从音频文件中提取音频特征。该函数支持提取多种不同的特征，
    包括MFCC、Chroma、MEL Spectrogram Frequency、Contrast和Tonnetz。
    函数采用音频文件的路径作为输入，并可选地指定要提取的特征类型。
    该函数的docstring提供了使用示例和支持的特征列表，这使得使用该函数变得非常容易。
    它还使用了Python的关键字参数（kwargs），在函数调用时，允许用户根据需要选择要提取的特征类型。
    在函数的实现中，它首先检查音频文件的格式是否正确，如果不正确，则将其转换为16000采样率和单声道通道。
    然后，它使用Librosa库提取所选的特征，并将它们连接成一个numpy数组，并返回该数组。

    这段代码使用了Python中的with语句和soundfile库中的SoundFile类。
    它的作用是打开名为file_name的音频文件，并将其作为sound_file对象传递给代码块，
    以便在代码块中对该文件进行操作。
    with语句的好处是，在代码块结束时，它会自动关闭文件句柄，无需手动关闭。
    使用soundfile.SoundFile()函数创建的sound_file对象是一个上下文管理器，它提供了一些方法和属性，
    可以用于读取和操作音频文件。在该函数中，我们使用sound_file对象读取音频文件，获取其采样率和数据类型等信息。
    在代码块的最后，with语句自动关闭了sound_file对象，释放了与该文件的所有资源。
    需要注意的是，在使用soundfile库打开音频文件时，我们可以使用with语句来确保文件句柄在使用完毕后被正确关闭。
    这可以避免在操作大量音频文件时出现资源泄漏和文件句柄耗尽等问题。

    params:
    -
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)

    examples:
    -
    >>> features = extract_feature(path, f_config)
    """
    # 抽取关键字参数中的bool开关信息
    # mfcc = kwargs.get("mfcc",True)
    # chroma = kwargs.get("chroma",True)
    # mel = kwargs.get("mel",True)
    # contrast = kwargs.get("contrast",False)
    # tonnetz = kwargs.get("tonnetz",False)


    if not set(f_config) <= set(ava_features):
        raise ValueError(f"{f_config}包含不受支持的特征名字,请在{ava_features}中选取")

    try:
        # print(audio_file_name,"@{audio_file_name}")
        #考虑将此时的工作路径切换为项目根目录,以便利用相对路径访问文件
        # os.chdir(project_dir)
        p = Path(audio_file_name)
        if p.is_file()==False:
            raise FileNotFoundError(f"{p.absolute().resolve()} does not exist")
        with soundfile.SoundFile(audio_file_name) as sound_file:
            # 成功打开
            pass
    except RuntimeError:
        # not properly formated, convert to 16000 sample rate & mono channel using ffmpeg
        # get the basename
        basename = os.path.basename(audio_file_name)  # name.ext
        dirname = os.path.dirname(audio_file_name)
        name, ext = os.path.splitext(basename)
        new_basename = f"{name}_c.wav"  # c表示converted
        new_filename = os.path.join(dirname, new_basename)
        v = convert_audio(audio_file_name, new_filename)
        if v:
            raise NotImplementedError(
                "Converting the audio files failed, make sure\
                                       `ffmpeg` is installed in your machine and added to PATH."
            )
    else:
        # 转换成功
        new_filename = audio_file_name
    # 此时音频文件可以统一的方式提取
    return extract_features_handler(new_filename, f_config)

extractors_debug=None
def extract_features_handler(new_filename, f_config):
    """根据传入的开关提取对应的特征,并合并每个提取了的特征向量

    处理函数体的实现,这里是其他版本的实现:

    1.
    extractors = [
        (mfcc, mfcc_extract),
        (chroma, chroma_extract),
        (mel, mel_extract),
        (contrast, contrast_extract),
        (tonnetz, tonnetz_extract),
    ]

    for f, extractor in extractors:
        if f in f_config:
            if extractor in (chroma_extract, contrast_extract):
                res = extractor(sample_rate, stft)
            else:
                res = extractor(X, sample_rate)
            result = np.hstack((result, res))
    2.
        if mfcc:
            res = mfcc_extract(X, sample_rate)
            result = np.hstack((result, res))
        if chroma:
            res = chroma_extract(sample_rate, stft)
            result = np.hstack((result, res))
        if mel:
            res = mel_extract(X, sample_rate)
            result = np.hstack((result, res))
        if contrast:
            res = contrast_extract(sample_rate, stft)
            result = np.hstack((result, res))
        if tonnetz:
            res = tonnetz_extract(X, sample_rate)
            result = np.hstack((result, res))
    

    Parameters
    ----------
    f_config:list[str]
        需要提取的特征

    new_filename : path_like


    Returns
    -------
    ndarray
        提取结果(shape=(n,))
    """
    with soundfile.SoundFile(new_filename) as sound_file:
        X, sample_rate, extractors1, extractors2, stft = pre_calculate(f_config, sound_file)

        # 建立一个空数组来存储需要提取的特征
        result = np.array([])
        f_res=None
        for f in f_config:
            # print(f,extractors1.get(f),extractors2.get(f))

            if extractors1.get(f):
                f_res=extractors1[f](X, sample_rate)
            elif extractors2.get(f):
                f_res=extractors2[f](sample_rate, stft)
            # print(f_res.shape,f,"@{f_res.shape}")#type:ignore
            result = np.hstack((result, f_res))
            
        # print(result.shape)
    return result

def pre_calculate(f_config, sound_file):
    X = sound_file.read(dtype="float32")
    sample_rate = sound_file.samplerate
        # print(f'{sample_rate=}')
        # 根据参数情况,提取需要的情感特征
        # 对于chroma和constrast两种特征,计算stft的幅值矩阵(复数取模,实数化)
    from config.EF import chroma, contrast, mel, mfcc, tonnetz
    extractors1 = {mfcc: mfcc_extract, mel: mel_extract, tonnetz: tonnetz_extract}
    extractors2 = {chroma: chroma_extract, contrast: contrast_extract}

    stft = []
    if chroma in f_config or contrast in f_config:
        stft = stft_prepare(X)
    return X,sample_rate,extractors1,extractors2,stft


def stft_prepare(X):
    # mfcc=True if mfcc in f_config else False
    stft_raw = librosa.stft(X)
    stft = np.abs(stft_raw)
    # print(f'{stft_raw=},{stft_raw.shape=}')
    # print(f'{stft=},{stft.shape=}')
    # shape=(1025, 60)
    # print(f'{stft_raw.shape=},{stft.shape=}')
    return stft


def tonnetz_extract(X, sample_rate):
    har = librosa.effects.harmonic(X)
    ttz = librosa.feature.tonnetz(y=har, sr=sample_rate)
    ttzT = ttz.T
    tonnetz = np.mean(ttzT, axis=0)
    # loginfo
    # print(f'{har.shape=},{ttzT.shape=},{tonnetz.shape=}')
    return tonnetz


def mel_extract(X, sample_rate):
    # 注意,新版本的librosa.feature要求用关键字参数指明,位置参数的用法将被废弃
    ms = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    # 结果矩阵ms.shape=(n_mels,T),n_mels(mel频带)默认为128
    msT = ms.T
    mel = np.mean(msT, axis=0)
    # log:info
    # print(f"{ms.shape=},{msT.shape=},{mel.shape=}")
    # print(f'{msT=},{mel=}')
    return mel


def mfcc_extract(X, sample_rate):
    """
    1. 当调用librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)时，返回一个形状为(40, T)的二维数组，其中T表示输入音频信号的帧数。
    每一列代表一个音频帧的40个MFCC系数(每一行表示一个音频文件的T个帧的第i个mfcc系数(i=1,2,..,40)，这些系数已经进行了离散余弦变换（DCT）以减少维度并增强特征。
    MFCC系数在音频信号处理中非常有用，因为它们能够捕捉音频信号的许多特征，例如说话人的身份、音调、声音的强度、语速等等。在语音识别中，MFCC系数通常被用来训练模型，以识别不同的语音单元（如音素）。

    2. 将mfcc矩阵进行转置可以使得每一行表示一个MFCC系数向量，每一列表示这个音频帧中对应的MFCC系数的值。
    这个操作可以使得MFCC系数更容易被用于进一步的数据分析和可视化。例如，可以将转置后的MFCC系数矩阵用作特征向量来训练机器学习模型，或者将其用于可视化音频中不同频率带的能量分布。此外，在一些情况下，MFCC系数的转置也可以使得它们更容易与其他特征进行组合，以提高模型的性能。

    3. 借助np.mean进行降维
    提取音频特征时，计算每个MFCC系数的平均值可以提供一种方式来总结每个音频文件的特征。通常情况下，用平均值作为表示音频文件的特征向量可以使得模型的训练更加稳定，减少过拟合的风险。mfccs.shape=(40,)

    Parameters
    ----------
    X : ndarray
        音频采样序列
    sample_rate : int
        采样率

    Returns
    -------
    ndarray
        计算得到的mfcc特征
    """
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    # shape=(40,60)
    mfccT = mfcc.T  # shape=(60,40)
    mfccs = np.mean(mfccT, axis=0)

    return mfccs


def contrast_extract(sample_rate, stft):
    ctr = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    ctrT = ctr.T
    contrast = np.mean(ctrT, axis=0)
    return contrast


def chroma_extract(sample_rate, stft):
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_stftT = chroma_stft.T
    chroma = np.mean(chroma_stftT, axis=0)
    # print(f'{chroma_stft=},{chroma_stft.shape=}')
    # print(f'{chroma_stftT=},{chroma_stftT.shape=}')
    # print(f'{chroma=},{chroma.shape=}')
    # print(f'{chroma_stft.shape=},{chroma_stftT.shape=},{chroma.shape=}')
    return chroma


def best_estimators(classification_task=True,fast=True):
    """
    从grid目录中读取经过计算的各个模型对应的最优超参数
    注意在做超参数搜索前是没有可用的文件可供读取,需要通过grid_search调用本类
    实例进行计算后才有的用

    其中,GradientBoostingRegressor比较耗时(1min左右才能计算出来)
    其他模型只要若干秒

    params:
    -
    classification:bool
        True:分类任务,读取各个分类器及其最优超参数
        False:回归任务,读取各个回归模型及其最优超参数
    """
    if classification_task:
        res= load(bclf)
    else:
        res= load(brgr)
    if fast:
        for i,estimator_tuple in enumerate(res):
            # print(estimator.__class__.__name__)
            estimator,_,_ = estimator_tuple
            if 'GradientBoosting' in estimator.__class__.__name__:
                res.remove(res[i])
    return res

def test1():

    
    audio_path= speech_dbs_dir/"emodb/wav/03a01Fa.wav"
    print(os.path.exists(audio_path))

    features = extract_feature_of_audio(audio_path, f_config_def)
    return features

if __name__ == "__main__":
    pass
    # res = get_audio_config(audio_config)
    # print(res)
    # res=best_estimators()


    audio_path= speech_dbs_dir/"emodb/wav/03a01Fa.wav"
    print(os.path.exists(audio_path))

    features = extract_feature_of_audio(audio_path, f_config_def)

