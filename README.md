[toc]



# Cross-corpus speech emotion recognition

## Introduction

- 跨库语音情感识别系统设计与实现
- 本项目采用的算法大多是传统的机器学习算法,实验表明,这些算法在单库识别上具有不错的性能和效果
- 采用深度学习的方法可以进一步提高识别性能
- 然而对于跨库识别而言,传统的机器学习算法表现的有些力不从心,近几年利用深度学习的方法对跨库识别的研究成为了情感计算的新热点

## 语音识别vs语音情感识别vs跨库语音情感识别

为了解释本文为何没有采用语音识别技术，需要先明确以下基本概念：

1. 语音识别：语音识别是一种将人类语音转换为计算机可理解的文本的技术。它的主要目标是识别说话者的语言内容。这项技术广泛应用于智能语音助手、语音搜索和自动语音转录等领域。
2. 语音情感（情绪）识别：语音情感识别关注的是分析说话者的情感状态，例如愤怒、快乐、悲伤等。这需要对声音的音调、语速、语调等特征进行分析。语音情感识别可以应用于客户服务、心理健康评估和人机交互等领域。为了提高识别性能，近年来许多关于情感的深度特征被研究出来应用于语音情感识别。
3. 跨语料库语音情感识别：这是一种在多种语言和文化背景下进行情感识别的技术。它需要考虑不同语言和文化背景下的语音特征和情感表达方式的差异。

本课题并未采用语音识别技术，因为语音中包含的文本信息不一定能准确表达语音的情感信息，尤其是短语音中的短文本，不适合用于情感识别。例如：甲对乙做恶作剧，乙向丙诉苦的时候说“甲一直笑个不停“，这是时候的语音情感和可能是愤怒的，但是仅从语音文本内容来看不容易判断出这句话是愤怒的。

### 异同点

-  语音识别和语音情感识别的主要区别在于关注点不同。语音识别关注语言内容，而语音情感识别关注情感状态。

-  语音情感识别和跨语料库语音情感识别的主要区别在于应用范围。语音情感识别通常针对单一语言或文化背景，而跨语料库语音情感识别需要处理多种语言和文化背景下的情感识别问题。并且通常是在单个库（而非混合库）上训练识别模型，在其他库上进行语音情感识别任务。

## Requirements

- 在conda环境中(推荐使用全新创建的环境)
  - 根据我的经验来看,我们只用conda创建环境,以及按照python
  - 但是安装第三方包的时候`conda install`往往不是一个好的选择
  - 如果不是包官方推荐,那么建议都用`pip install`安装
    - 例如pytorch支持用conda安装
    - tensorflow推荐用pip 安装
  - 特别是当包的依赖关系复杂,每次安装要分析很久,当项目依赖的包版本都清楚可用,那么意味着检查依赖这一步不需要了,使用`pip install`会高效的多
- 安装python`conda install python=3.9`
- 执行安装命令行:`pip install -r requirements.txt`

### python version

- **Python 3.9**

### Python Packages

- 可执行文件ffmpeg

### 安装环境和依赖的补充说明🎈

- ```bash
  pip3 install -r requirements.txt
  ```

- 通常,越是基础的库越应该放在后面安装,例如librosa的可能会依赖于特定版本而不是最新版本的基础库(例如numpy,matplotlib)

- 而一些可以独立工作的包安装位置相对不那么严格

- 创建环境的时候注意带上python,或者创建一个空环境后,先使用`conda install python=3.9`来安装python,否则后续的`pip install`将无法正常工作

- python的版本受到许多依赖的限制,尤其是深度学习框架(比如`tensorflow`),初次之外,python的版本可以根据需要调整,为了兼容性起见,通常推荐python的版本举例最新发布的python版本低2到3个中版本,例如当前的最新版本是`3.m.n`那么推荐那安装`3.(m-2)`,最后的n通常无关紧要,在这种情况下一些第三方包一般都适配到了`3.(m-n)`

#### 需要注意的包

- librosa 
  - librosa 0.9.2 is not the lastes version,but the newest version don't work well with some matplotlib version
  - these problems appeared that I install `matplotlib` with `conda install` and `librosa`with `pip isntall`
  - The compatibility issues may be caused by mixing two installation methods.
  - so I just use 0.9.2 version instead of the lastest one
  - However, the newer version may become the more preferred choice in the future, once the bugs or compatibility issues have been fixed.

- pluggy
  - the pluggy may installed automatically or not(I write here becases it when I test the `requirements.txt` in a brand new conda environment ,the pip prompt me that the pluggy was not installed 

- tensorflow
  - if you just want to experience the basic ML alogrithms' working on SER task,it's no need for you to install tensorflow
  - in may case , I use tensorflow==2.10,but other version of tensorflow above 2.6 may work well too


### 其他情况说明

- 我在一次调试matplotlib backend设置的过程中,偶然发现在notebook中使用`%matplotlib qt`失败

- 后来我尝试卸载重装matplotlib和pyside

- 然而提示我一些`dll`问题和权限错误问题

- 这些问题时平时不曾遇到

- 我猜测可能时机器太久没有关机了(平时我都是休眠),导致系统出现了一些错误

  - 机器发生错误是很有可能的,就比如学校图书馆的刷脸系统,`验证成功`平时的话屏幕显示绿色提醒,然而最近`验证成功`显示的也是红色
  - 而在早期的windows7上,有时候从休眠中回复直接会失败

- 然后我重启机器,发现任务栏多出了个搜索框,系统更新了没有重启可能也造成了一些影响

- 有时候还会出现各种意外的问题

  - 例如`KNeighborsClassifier`训练完毕后,用它预测新的样本或在训练集上预测会报错:

    - ```python
      
      File d:\condaPythonEnvs\tf210\lib\site-packages\sklearn\neighbors\_classification.py:237, in KNeighborsClassifier.predict(self, X)
          235     neigh_dist = None
          236 else:
      --> 237     neigh_dist, neigh_ind = self.kneighbors(X)
          239 classes_ = self.classes_
          240 _y = self._y
      
          643 get_config = getattr(self._dynlib, "openblas_get_config",
          644                      lambda: None)
          645 get_config.restype = ctypes.c_char_p
      --> 646 config = get_config().split()
          647 if config[0] == b"OpenBLAS":
          648     return config[1].decode("utf-8")
      
      AttributeError: 'NoneType' object has no attribute 'split'
      ```

    - 而当我切换到另一个环境又可以工作,说明是应该不是系统问题,而更可能是环境出了问题

#### 意料之外的错误的解决方案

- 重启你的机器,这或许解决一些系统错误
- 创建/切换虚拟环境,也许是安装某些包引起依赖降级或者环境污染

#### 关于进度条显示:tqdm

- `tqdm` 是一个Python的进度条库，用于在控制台中展示循环的进度，并可以根据已经完成的工作量估计剩余时间。它的使用非常简单，以下是一些常见的用法：
- `tqdm` derives from the Arabic word *`taqaddum`* (تقدّم) which can mean "progress," and is an abbreviation for "I love you so much" in Spanish (*te quiero demasiado*).

- 在循环中使用tqdm来展示进度条：

```python
from tqdm import tqdm
import time

for i in tqdm(range(10)):
    time.sleep(0.1)
```

这将展示一个进度条，表示在循环中的迭代次数。在每个迭代中，进度条将更新并显示已经完成的进度百分比。

- 可以通过设置`desc`参数来为进度条添加描述信息：

```python
from tqdm import tqdm
import time

for i in tqdm(range(10), desc="Processing data"):
    time.sleep(0.1)
```

这将在进度条上方添加一个描述信息，描述当前正在执行的任务。

- 可以通过设置`leave`参数来控制进度条是否应该在循环结束后保留：

```python
from tqdm import tqdm
import time

for i in tqdm(range(10), leave=False):
    time.sleep(0.1)
```

这将在循环结束后清除进度条。

这些是`tqdm`的一些常见用法，它还具有许多其他有用的功能，如支持多线程和多进程，可以自定义进度条的样式等。

- 本项目中遍历`estimators`

- ```python
  from tqdm import tqdm
  from time import sleep
  from audio.core import best_estimators
  ests=best_estimators()
  ests=tqdm(ests)
  for x in ests:
      sleep(0.5)
      print(x)
  ```

  

### PYTHONPATH环境变量

- 为了放便地运行本项目,请配置环境变量`PYTHONPATH`,并至少向其中添加本项目的根目录

- 例如,我的项目放在`$ccser=D:\repos\CCSER\SER`,根据自己的clone/download位置情况修改这个值

- 例如,在使用powershell配置(windows系统下)

  - ```powershell
    $ccser="D:\repos\CCSER\SER"
    setx PYTHONPATH "$ccser;$env:PYTHONPATH"
    ```

  - 执行:

    - ``` powershell
      PS C:\Users\cxxu> $ccser="D:\repos\CCSER\SER"
      PS C:\Users\cxxu> $ccser
      D:\repos\CCSER\SER
      
      PS D:\repos\configs\env> setx PYTHONPATH "$ccser;$env:PYTHONPATH"
      
      SUCCESS: Specified value was saved.
      ```

## SpeechDatabases

- 这里主要使用3个语音数据库

### RAVDESS

- [**RAVDESS**](https://zenodo.org/record/1188976) : The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.
- [RAVDESS Emotional speech audio | Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?resource=download)
- **File naming convention**

  Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

  *Filename identifiers*

  - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  - Vocal channel (01 = speech, 02 = song).
  - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
  - Repetition (01 = 1st repetition, 02 = 2nd repetition).
  - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

- *Filename example: 03-01-06-01-02-01-12.wav*

  1. Audio-only (03)
  2. Speech (01)
  3. Fearful (06)
  4. Normal intensity (01)
  5. Statement "dogs" (02)
  6. 1st Repetition (01)
  7. 12th Actor (12)
     Female, as the actor ID number is even.

- RAVDESS语料库（Ryerson Audio-Visual Database of Emotional Speech and Song）是一个包含了人类语音和歌曲记录的数据库。该数据库包含了24名演员在读出短语时表现出八种情感状态的语音记录，以及12首歌曲的音频记录。

  RAVDESS语料库的语音记录包含了两种语言（英语和法语），以及四种情感状态的强度（高、中、低和中性）。情感状态包括愤怒、厌恶、恐惧、快乐、悲伤、惊讶和中性。每个演员都会读出两个句子，每个句子表达了四种不同的情感状态。每个短语的长度为三到五个单词。RAVDESS语料库的音频文件格式为WAV，采样率为48kHz，16位量化。

  RAVDESS语料库的歌曲记录包含了12首歌曲，每首歌曲都表达了四种不同的情感状态，包括快乐、悲伤、惊讶和中性。每首歌曲的长度为30秒至1分钟不等，音频文件格式为MP3。

  RAVDESS语料库是一个广泛应用于语音情感识别和分类领域的标准数据集，它已经被广泛应用于语音情感识别和分类算法的开发和评估。该数据库的开放访问使得研究人员可以更方便地进行情感识别和分类算法的开发和评估，同时也为智能语音应用的开发提供了有用的资源。

###  SAVEE

#### Speakers

'DC', 'JE', 'JK' and 'KL' are four male speakers recorded for the SAVEE database


--------------------------------------

#### Audio data 

Audio files consist of audio WAV files sampled at 44.1 kHz

There are 15 sentences for each of the 7 emotion categories.

The initial letter(s) of the file name represents the emotion class, and the following digits represent the sentence number.

- The letters 'a', 'd', 'f', 'h', 'n', 'sa' and 'su' represent 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise' emotion classes respectively. 

- E.g., 'd03.wav' is the 3rd disgust sentence. 

### EMODB

- [**EMO-DB**](http://emodb.bilderbar.info/docu/) : As a part of the DFG funded research project SE462/3-1 in 1997 and 1999 we recorded a database of emotional utterances spoken by actors. The recordings took place in the anechoic chamber of the Technical University Berlin, department of Technical Acoustics. Director of the project was Prof. Dr. W. Sendlmeier, Technical University of Berlin, Institute of Speech and Communication, department of communication science. Members of the project were mainly Felix Burkhardt, Miriam Kienast, Astrid Paeschke and Benjamin Weiss.
- [EmoDB Dataset | Kaggle](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb?resource=download)
- EMODB是爱丁堡多情感数据库（Edinburgh Multi-Emotion Database）的缩写，是一个包含了由演员表演不同情感的音视频记录的数据库。它由爱丁堡大学的研究人员创建，旨在支持情感识别和分类算法的开发和评估。

  该数据库包含了来自英国的10位专业演员（5男5女）的535个音视频记录。每位演员表演了12种不同的情感，包括愤怒、厌恶、恐惧、快乐、悲伤、惊讶等等。这些记录是在一个标准化的环境中进行的，包括标准化的灯光、背景和摄像机角度。

  该数据库已广泛用于情感识别和分类等领域的研究，以及其他相关领域，如语音处理、情感计算和人机交互。该数据库可免费供学术研究使用。
- Code of emotions:

  | letter | emotion (english) | emotion (german) | letter              |
  | ------ | ----------------- | ---------------- | ------------------- |
  | W      | anger             | Ärger (Wut)      | A                   |
  | L      | boredom           | Langeweile       | B                   |
  | E      | disgust           | Ekel             | D                   |
  | A      | anxiety/fear      | Angst            | F                   |
  | F      | happiness         | Freude           | H                   |
  | T      | sadness           | Trauer           | S                   |
  |        |                   |                  | N = neutral version |

- EMODB是一个包含了演员表演不同情感的音视频记录的数据库，其中语音文件的命名方式比较规范，以下是一个示例文件名的分析：

  03a01Wa.wav

  - 03 表示这个音频记录来自第3位演员
  - a01 表示这个音频记录是该演员表演的第1种情感
  - W 表示这个情感是“愤怒”（Angry）的缩写
  - a 表示这个是该情感的第1个副本（第一个表演）
  - .wav 表示这个文件的格式为.wav格式

  因此，这个文件名告诉我们，这个音频记录来自EMODB数据库中的第3位演员，表演的是愤怒情感，并且这是该演员表演愤怒情感的第1个副本。文件的格式为.wav格式。EMODB的语音文件命名方式比较规范，这些信息对于进行情感识别和分类等研究非常有用。
- Additional Information

  Every utterance is named according to the same scheme:

  - Positions 1-2: number of speaker
  - Positions 3-5: code for text
  - Position 6: emotion ( letter stands for german emotion word)
  - Position 7: if there are more than two versions these are numbered a, b, c ....

  Example: 03a01Fa.wav is the audio file from Speaker 03 speaking text a01 with the emotion "Freude" (Happiness).

### Kaggle社区

- Kaggle是一个由Google收购的在线数据科学和机器学习社区，它提供了各种数据科学竞赛、数据集、内置工具和协作平台。Kaggle的目标是促进数据科学和机器学习领域的发展，让数据科学家和机器学习专家可以在这个平台上分享和交流各种数据科学工具、技术和最佳实践。

  Kaggle提供了丰富的数据集和挑战，在这些挑战中，数据科学家和机器学习专家可以使用各种技术和算法来解决各种问题和挑战，比如预测股票价格、图像分类、自然语言处理等。Kaggle还提供了一个协作平台，让数据科学家和机器学习专家可以在这个平台上分享代码、数据和最佳实践，并且可以互相学习和协作。

  Kaggle已经成为了数据科学和机器学习领域的一个重要社区，它不仅为数据科学家和机器学习专家提供了丰富的资源和工具，也为企业和组织提供了一种新的方式来解决各种数据科学和机器学习领域的问题。

### 添加其他语料库到本项目

- 添加语料库(记为db)到本项目中比较简单,只需要在config模块中执行一定的配置即可
- 不过由于语料库的命名规范的不同,您或许要亲自编写针对于db的`create_{db}_meta()`函数,



## 语料库文件在项目中的组织与划分

- 将EMODB语料库放在单独的目录`data/emodb`
- 将RAVDESS语料库
  - 大部分样本放在了训练集目录`data/ravdess`

- 测试集(验证集)目录`data/validation`

#### powershell递归统计脚本:

- `ls -r -File |measure|select Count`

- ```powershell
  $path=".";
  Get-ChildItem $path -Directory -Recurse | ForEach-Object {
       $count = (Get-ChildItem $_.Name -File).Count
       Write-Output "$($_.Name): $count files"
  }
  ```

  

#### 文件统计结果



- 各个子目录的文件统计

- `ls  |%{$_;(ls $_| measure)|select count}`可以统计子目录的文件数


### desc_files(csv 元数据文件)🎈

- 由于不同语料库的文件名规范不同,所以在使用前应该进行基本的统一处理(主要抽取语音文件路径和文件的情感标签)
- 可以利用pandas将抽取结果持久化保存到磁盘,以便后续使用
  - 特别是对于大量文件来说,保存处理结果有利于提高重复试验的效率

- desc_files (csv)文件存储的是语料库的语音文件的路径(path),情感类别(emotion),存放在项目根目录的`meta_files`目录下

### Emotions available@情感类别

- 本项目采用的语料库中包含 "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "surprise","ps" (pleasant surprise) and "boredom".
- 尽管如此,您不总是可以任意搭配情感组合,因为有些语料库缺失其中的若干种情感
- 比较常见的组合包括:
  - angry,happy,sad
  - angry,happy,sad,neutral,pleasant


## Feature Extraction@特征提取🎈

- 
  特征提取是语音情感识别系统的主要部分。它基本上是通过将语音波形转换为参数形式的表示形式，以相对较低的数据速率完成的。

- “数据速率("Data rate")”，它是指在特定时间内传输的数据量。在语音情感识别系统中，数据速率是指每秒钟传输的语音数据量，通常以比特率（bits per second）或千位每秒（kilobits per second）为单位进行度量。需要注意的是，数据速率还可以用于描述其他类型的数据传输，例如网络传输或存储介质的读取速度。在这些情况下，它通常指在特定时间内传输或处理的数据量，通常以比特率或字节率（bytes per second）为单位进行度量。

- 特征提取的过程通过将语音波形转换为参数形式的表示形式，可以减少语音信号的数据速率。这是因为，原始语音信号通常包含大量冗余信息，而通过提取与情感状态相关的声学特征，可以压缩数据并减少传输所需的带宽和存储空间。

- 因此，特征提取对于高效处理和分析大量语音数据是至关重要的。


### 常见的语音情感特征

在语音情感识别中，常用的特征包括以下几种：

1. 频谱特征：频谱特征包括共振峰频率、谐波比、频谱能量等。这些特征反映了声音的频域信息，对于识别不同情感状态具有很好的区分能力。
2. 韵律特征：韵律特征包括基频、语速、节奏等。这些特征反映了声音的时间和节奏信息，对于识别语音情感状态也非常重要。
3. 非线性特征：非线性特征包括声调、颤音、嘶哑等。这些特征反映了声音的质量和表现形式，对于识别不同情感状态也有很好的识别能力。
4. 语音质量特征：语音质量特征包括噪声、失真、清晰度等。这些特征反映了语音信号的质量，可能对情感状态的识别产生影响。
5. 情感词汇特征：情感词汇特征是从语音信号中提取出与情感状态相关的词汇，可以通过情感词典等工具来实现。

- 这些特征在不同情感状态之间具有不同的区分能力，因此在设计语音情感识别系统时需要综合考虑它们的优缺点，并根据具体情况选择合适的特征组合。例如，一些研究表明，基于频谱特征和韵律特征的特征组合可以取得较好的情感识别效果；而其他研究则发现，非线性特征对于识别某些情感状态具有更好的区分能力。因此，在实际应用中，需要根据具体情况选择适合的特征组合，并通过机器学习算法等手段对语音信号进行分类和识别。

### 本项目可选的情感特征

- [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- Chromagram
- Contrast
- MEL Spectrogram Frequency (mel)
- Tonnetz (tonal centroid features)

### 特征预处理

- 对于语音信号的情感识别任务，MFCC、MEL、Contrast 等特征通常是最常用的特征。在使用这些特征进行建模之前，可以考虑进行以下预处理：

  1. 归一化：对于不同的特征，可能具有不同的取值范围，例如 MFCC 和 MEL 系数具有不同的幅度范围。因此，可以对所有特征进行归一化或标准化，以确保它们具有相同的尺度。
  2. 帧级别的能量归一化：语音信号在录制过程中可能会存在噪声或音量变化等问题，这可能会导致特征的幅度变化。因此，可以对每个帧的能量进行归一化，以确保在不同的录音条件下，特征的幅度保持一致。
  3. 去除静音部分：对于包含静音的语音信号，可以通过阈值或其他语音活动检测算法来去除静音部分，以减少噪声的影响。
  4. 数据增强：可以通过变换语音信号的速度、音量、音调等方式，生成更多的训练数据，以提高模型的鲁棒性和泛化能力。
  5. 特征选择：可以使用特征选择算法，如相关系数或基于模型的方法，来选择最相关的特征，以减少特征的数量和计算成本。

  总之，在使用 MFCC、MEL、Contrast 等特征进行语音情感识别任务之前，可以对数据进行归一化、能量归一化、去除静音部分、数据增强和特征选择等预处理，以获得更好的模型性能。

  1. 归一化

  可以使用 sklearn 中的 StandardScaler 对特征进行归一化，代码如下：

  ```python
  from sklearn.preprocessing import StandardScaler
  
  # X为特征矩阵，axis=0对每列进行归一化
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  ```

  1. 帧级别的能量归一化

  可以使用 librosa 中的 power_to_db 函数计算每个帧的能量，然后使用 sklearn 中的 MinMaxScaler 对能量进行归一化，代码如下：

  ```python
  from sklearn.preprocessing import MinMaxScaler
  import librosa
  
  # y为语音信号，sr为采样率
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
  log_S = librosa.power_to_db(S, ref=np.max)
  
  # 计算每个帧的能量
  frame_energy = np.sum(np.exp(log_S), axis=0)
  
  # 对帧级别的能量进行归一化
  scaler = MinMaxScaler()
  frame_energy = scaler.fit_transform(frame_energy.reshape(-1, 1)).reshape(-1)
  ```

  1. 去除静音部分

  可以使用 librosa 中的 amplitude_to_db 函数将语音信号转换为分贝表示，然后使用 librosa 中的 onset_detect 函数检测语音活动部分，代码如下：

  ```python
  import librosa
  
  # y为语音信号，sr为采样率
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
  log_S = librosa.power_to_db(S, ref=np.max)
  
  # 将语音信号转换为分贝表示
  db_S = librosa.amplitude_to_db(S, ref=np.max)
  
  # 检测语音活动部分
  onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
  onset_times = librosa.frames_to_time(onset_frames, sr=sr)
  ```

  1. 数据增强

  可以使用 librosa.effects 中的 time_stretch、pitch_shift 和 dynamic_range_compression 函数，对语音信号进行时间拉伸、音高变换和动态范围压缩，代码如下：

  ```python
  import librosa
  
  # y为语音信号，sr为采样率
  y_stretch = librosa.effects.time_stretch(y, rate=0.8)
  y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)
  y_drc = librosa.effects.dynamic_range_compression(y, threshold=1.0, ratio=4.0)
  ```

  1. 特征选择

  可以使用 sklearn 中的 SelectKBest 和 mutual_info_classif 函数，选择与情感识别任务最相关的 k 个特征，代码如下：

  ```python
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import mutual_info_classif
  
  # X为特征矩阵，y为情感标签
  selector = SelectKBest(mutual_info_classif, k=10)
  X_new = selector.fit_transform(X, y)
  ```

  以上是一些在使用 MFCC、MEL、Contrast 等特征进行语音情感识别任务时的代码建议，希望能对你有所帮助。

### 补充

* MFCC：MFCC是一种常用的声学特征，它是通过将语音信号转换为梅尔频率倒谱系数来提取的。MFCC具有对于语音信号中的频率变化较为敏感的特点，对于识别不同情感状态具有很好的区分能力。
* Chromagram：Chromagram是一种基于音高的特征，它通过计算语音信号中不同的音高分布来提取。
* Contrast对比度：对比度是一种基于语音信号的强度变化的特征，它可以通过计算不同频率分量之间的能量差异来提取。对比度对于识别语音信号中的强度变化具有很好的区分能力，但在某些情况下可能对于情感状态的识别不够敏感。
* MEL频谱：MEL频谱是一种基于人耳听觉模型的特征，它将语音信号转换为梅尔频率分布，并对每个频率分布进行离散余弦变换以提取特征。MEL频谱对于语音信号中的频率变化较为敏感，但与MFCC相比，它在某些情况下可能具有更好的区分能力。
* Tonnetz：Tonnetz是一种基于音高的特征，它通过计算不同音高之间的距离和组合关系来提取。在律学与和声学中，调性网络，或托内斯（来自于德语“Tonnetz”，“tone-network”的意思）是一种用于表示调性空间的、概念性的音乐格子图，由莱昂哈德·欧拉于1739年提出。调性网络的各种可视化形式可被用于表示欧洲古典音乐的传统和声关系。Tonnetz在音乐信息检索和音乐情感识别中广泛应用.
* 总的来说，这些特征在语音情感识别中都具有一定的应用价值，但具体选用哪些特征需要根据实际应用情况和数据分析结果进行选择。在特征提取的过程中，需要综合考虑不同特征之间的互补性和差异性，并对特征进行合理的组合和调整，以提高语音情感识别的准确率和鲁棒性。

### features目录下npy文件

- 其中test开头的文件是少量语音文件提取的特征保存文件,可以用来调试代码
- train开头得到文件是大量的文件提取出来的特征文件
- 他们的名称结构是`{partition}_{feature_label}_{first_letters}_{n_samples}.npy`
  - partition$\in${'test','train'}
  - feature_label表示提取了哪些特征(特征链,用`-`连接不同特征)
  - first_letters表示情感种类(单词的首字母)
    - 为了控制数据大小,预设2种情感模式(HNS和AHNPS)

## Grid Search@网格搜索

- [3.2. Tuning the hyper-parameters of an estimator — scikit-learn documentation](https://scikit-learn.org/stable/modules/grid_search.html)
- 网格搜索是一种机器学习中的超参数调优技术，其目的是找到模型超参数的最优值。
- 超参数是在训练过程中不会被学习的参数，但在训练前需要设置，可以对模型性能产生重要影响。
- Grid Search 可以翻译为“网格搜索”或“网格调参”，是一种常用的机器学习超参数调优方法。
- Grid Search 的目的是通过遍历给定的超参数组合，找到最优的超参数组合，以获得最佳的模型性能。它基于一个预定义的超参数网格（grid），对每个超参数组合进行评估和比较，从而选择最佳的超参数组合。
- 网格搜索涉及 <u>定义一个超参数值的矩阵</u>，并且系统地搜索矩阵以寻找在验证数据集上表现最佳的超参数组合。
- 具体来说，Grid Search 将每个超参数的取值范围划分成一组离散的值，然后对所有可能的超参数组合进行遍历，对每个组合训练一个模型，并使用交叉验证等方法评估模型性能。
- 最后，选择具有最佳性能的超参数组合作为最终模型的超参数。
- Grid Search 是一种简单而有效的调参方法，但它需要遍历所有可能的超参数组合，因此计算成本较高。为了减少计算成本，可以使用 <u>随机搜索（Random Search）</u>等其他调参方法。



### 计算best_model

- `grid.search`模块提供了计算各个算法最优超参数的实现

# 本项目使用的算法(分类模型)😂

## ML Classifiers

- SVC
- RandomForestClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- MLPClassifier
- BaggingClassifier
- Recurrent Neural Networks (Keras)

### 选择合适的分类器(用k-fold评估)

- ```python
  from sklearn.datasets import load_iris
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import cross_val_score
  
  # 加载iris(鸢尾花)数据集
  X, y = load_iris(return_X_y=True)
  
  # 使用线性回归模型进行交叉验证
  model = LinearRegression()
  
  scores = cross_val_score(model, X, y, cv=5)
  print("Scores:", scores)
  print("Mean score:", scores.mean())
  ```

  - ```bash
    Scores: [0.96666667 0.96666667 0.9        0.96666667 1.        ]
    Mean score: 0.9600000000000002
    ```

- ```python
  # 加载iris(鸢尾花)数据集
  X, y = load_iris(return_X_y=True)
  
  # model=RandomForestClassifier()
  model=SVC()
  scores = cross_val_score(model, X, y, cv=5,verbose=3)
  print("Scores:", scores)
  print("Mean score:", scores.mean())
  ```

  - ```bash
    Scores: [0.96666667 0.96666667 0.96666667 0.93333333 1.        ]
    Mean score: 0.9666666666666666
    ```

- 而使用决策树模型进行分类,效果比较好,在5折叠验证中,都在0.9以上

- 使用SVC或者RF,效果更加好

#### 使用shuffle

- 不对有序数据集iris洗牌,效果:

  ```python
  
  # 加载iris(鸢尾花)数据集
  X, y = load_iris(return_X_y=True)
  
  # 定义5折交叉验证
  kf = KFold(
      n_splits=5,
      #    shuffle=True,
      # random_state=42,
  )
  
  # 使用线性回归模型进行训练和测试
  model = LinearRegression()
  # model=RandomForestClassifier()
  scores_cv = []
  for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      model.fit(X_train, y_train)
      score = model.score(X_test, y_test)
      scores_cv.append(score)
      print("Score:", score)
  mean_score = np.mean(scores_cv)
  print(f"{mean_score=}")
  ```

  - ```bash
    Score: 0.0
    Score: 0.8512492308414581
    Score: 0.0
    Score: 0.7615543936085848
    Score: 0.0
    mean_score=0.32256072489000853
    ```

  - 通常出现这种情况的话,可以认为是数据集读入策略有问题

- 解开shuffle=True的注释,得到正常的预测性能

- ```bash
  Score: 0.9468960016420045
  Score: 0.9315787260143983
  Score: 0.9177129838664249
  Score: 0.9026578332122843
  Score: 0.921073136533955
  mean_score=0.9239837362538135
  ```

  

#### SVC

- Scikit-learn中的SVC是一种支持向量机（Support Vector Machine）分类器，用于解决二分类和多分类问题。SVC是一种非常强大的模型，可以处理高维度的数据，并且能够有效地处理非线性可分的数据。

- SVC的主要思想是在特征空间中找到一个最优的超平面（hyperplane），将不同类别的数据分开。
- 在二维空间中，超平面是一条直线，而在高维空间中，超平面是一个超平面。
- SVC的训练过程是通过寻找一个**最大间隔**（maximum margin）的超平面来实现的，即找到一个超平面，使得所有训练样本离该超平面的距离最大化。这个最大间隔的超平面是通过拉格朗日乘子法（Lagrange multipliers）求解一个二次规划（quadratic programming）问题来实现的。
- SVC可以使用不同的**核函数**（kernel function）来学习非线性的决策边界。
  - 常用的核函数包括线性核函数、多项式核函数、径向基函数（Radial Basis Function，RBF）核函数等。
  - 这些核函数可以将低维空间中的数据映射到高维空间中，从而使得<u>数据在高维空间中变得线性可分</u>。
- SVC还具有一些重要的超参数，包括C、kernel、gamma等。
  - C是正则化参数，用于控制模型的复杂度和对训练数据的拟合程度。
  - kernel是核函数的选择，
  - gamma是径向基函数核的系数，它们都影响着模型的性能和复杂度。
- 在Scikit-learn中，SVC的使用非常简单，只需要创建一个SVC对象，设置一些超参数，然后调用fit()方法训练模型。可以使用predict()方法对新数据进行分类预测。
- 总之，SVC是一种强大的分类器，适用于解决二分类和多分类问题，尤其擅长处理高维度和非线性可分的数据。

##### sklearn.svm.svc

- [`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm).SVC

- SVC类是一种支持向量机分类器，用于二元和多元分类。
- 它的实现基于libsvm。拟合时间至少与样本数的平方成比例，并且在数万个样本之外可能是不切实际的。
- 对于大型数据集，请考虑使用LinearSVC或SGDClassifier，可能需要使用Nystroem转换器或其他核逼近。多类支持根据一对一方案处理。
- 有关提供的核函数的精确数学公式以及gamma、coef0和degree如何相互影响的详细信息，请参见叙述性文档中的相应部分：[kernel-function核函数](https://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation)。
- 在机器学习中，核函数是一种用于度量数据点之间相似性的方法，它可以将低维的非线性数据映射到高维的线性空间，从而使得一些线性算法，如支持向量机（SVM），能够处理非线性问题。然而，核函数的一个缺点是计算复杂度很高，尤其是当数据集很大时，因为需要计算每对数据点之间的核值。
- 为了解决这个问题，scikit-learn提供了一些核近似方法，它们可以用低维的特征向量来近似高维的核空间，从而降低计算成本和内存需求。
- `sklearn.kernel_approximation`模块包含一些函数，用于近似与某些内核（如支持向量机中使用的内核）相对应的特征映射。
  - 以下特征函数执行输入的非线性变换，可以作为线性分类或其他算法的基础。
  - 使用近似显式特征映射的优点与使用内核技巧相比，内核技巧隐式地使用特征映射，显式映射可以更适合在线学习，并且可以显著降低使用非常大的数据集进行学习的成本。标准的核化支持向量机不适用于大型数据集，但是使用近似核映射可以使用更高效的线性支持向量机。特别是，将核映射近似与 SGDClassifier 结合可以使大型数据集上的非线性学习成为可能。

##### 补充

- "Nystroem" 是一种用于解决大规模机器学习问题的方法，它是由计算机科学家 William N. Nystrom 在2002年提出的。该方法主要用于解决核矩阵的计算问题，可以大大减少计算复杂度和内存使用，从而使得处理大规模数据集的机器学习问题变得更加高效和可行。

  在传统的核方法中，需要计算样本之间的核矩阵，这个矩阵往往是比较大且密集的，计算和存储都会带来很大的困难。而 Nystroem 方法则是通过随机采样的方式来估计核矩阵，从而避免了计算和存储大型矩阵的问题。具体来说，该方法先从原始数据集中随机选择一部分样本，然后计算这些样本之间的核矩阵，再利用这个估计的核矩阵来近似计算全部样本之间的核矩阵。

  Nystroem 方法在实践中被广泛应用于各种机器学习问题，如分类、回归、聚类等。它可以提高处理大规模数据集的效率，同时保持较高的预测精度

- "SGDClassifier" 是 Scikit-learn （一个流行的 Python 机器学习库）中的一个分类器，它使用随机梯度下降算法来进行模型训练。

  随机梯度下降（Stochastic Gradient Descent，简称 SGD）是一种常用的优化算法，它可以在大规模数据集上进行快速的模型训练。SGDClassifier 利用 SGD 算法来最小化分类误差（或者其他损失函数），从而学习出一个分类模型。

  SGDClassifier 可以用于二分类问题和多分类问题，可以处理稀疏数据和高维数据。在使用 SGDClassifier 进行模型训练时，需要设置一些超参数，如学习率、正则化系数、损失函数等，这些超参数可以影响模型的性能和泛化能力。

  SGDClassifier 在实践中被广泛应用于各种分类问题，如文本分类、图像分类、音频分类等。它具有高效、可扩展、灵活等优点，是机器学习领域中常用的分类器之一。

##### 超参

- SVC（Support Vector Classification）是一种支持向量机分类器，其参数比较多且较为复杂，需要逐个进行解释。

  以下是SVC的参数及其含义：

  1. kernel：核函数类型。SVC支持多种核函数，包括线性核、多项式核、径向基函数（RBF）核等。默认值为'rbf'。
  2. degree：多项式核函数的次数。如果使用多项式核函数，需要指定degree参数。默认值为3。
  3. gamma：核函数的系数。当使用'RBF'、'poly'或'sigmoid'核函数时，需要指定gamma参数。较大的gamma值会导致决策边界更加复杂，对训练集的拟合程度更高，但容易出现过拟合。默认值为'scale'，表示使用1 / (n_features * X.var())作为gamma的值。
  4. coef0：核函数中的常数项。如果使用多项式核函数或sigmoid核函数，需要指定coef0参数。默认值为0。
  5. C：正则化参数。C控制了分类器的容错能力，即决策边界的软硬程度。较小的C值会导致决策边界更加平滑，较大的C值会导致决策边界更加复杂，对训练集的拟合程度更高。默认值为1.0。
  6. shrinking：是否使用缩小启发式。当数据集较大时，启用缩小启发式可以加快模型训练速度，但可能会影响模型性能。默认值为True。
  7. probability：是否启用概率估计。如果启用概率估计，在预测时会输出每个类别的概率估计值。默认值为False。
  8. tol：训练停止的容忍度。当模型的训练误差减少到tol时，训练过程停止。默认值为1e-3。
  9. max_iter：最大迭代次数。如果模型在指定的最大迭代次数内未收敛，则停止训练。默认值为-1，表示不限制迭代次数。
  10. class_weight：类别权重。如果数据集中的某些类别比其他类别更重要，则可以使用class_weight参数来平衡类别权重。默认值为None，表示所有类别的权重相等。
  11. verbose：输出详细信息的级别。默认值为0，表示不输出任何信息。
  12. cache_size：用于存储核矩阵的缓存大小。如果数据集较大，可以增大cache_size以提高模型训练速度。默认值为200MB。

##### tips

在使用SVC时，需要根据具体的问题选择合适的参数。以下是一些设置SVC参数的技巧：

1. 根据数据集的特征选择核函数：在选择核函数时，需要根据数据集的特征选择合适的核函数。对于线性可分的数据集，使用线性核函数可以获得较好的分类效果；对于非线性可分的数据集，可以尝试使用多项式核函数或径向基函数（RBF）核函数。
2. 调整正则化参数C：正则化参数C控制了决策边界的软硬程度。在模型欠拟合时，可以尝试增大C值；在模型过拟合时，可以尝试减小C值。
3. 调整核函数的参数：如果使用多项式核函数或RBF核函数，需要调整degree和gamma参数。在模型欠拟合时，可以尝试增大degree或gamma值；在模型过拟合时，可以尝试减小degree或gamma值。
4. 使用交叉验证选择最优参数：可以使用交叉验证来选择最优的参数组合，以获得最佳的分类效果。在训练模型时，可以将数据集分成多份，每次使用其中一份作为验证集，其余部分作为训练集，计算模型在验证集上的分类精度，并记录最佳的参数组合。

如果模型在使用合理的参数组合时仍然无法收敛，可以尝试以下技巧：

1. 增大正则化参数C：在模型欠拟合时，可以尝试增大正则化参数C，以增强模型的泛化能力。
2. 减小复杂度：可以尝试减小模型的复杂度，如减小多项式核函数的次数、减小RBF核函数的gamma值等。
3. 数据预处理：可以尝试对数据进行预处理，如归一化、标准化等，以提高模型的训练效果。
4. 增加训练时间：可以尝试增加模型的训练时间，以增加模型的训练次数，提高模型的分类精度。

需要注意的是，调整参数时需要谨慎，避免在训练集上出现过拟合。在使用SVC时，建议先使用默认参数进行训练，再根据实际情况进行调整。

#### RandomForestClassifier

- RandomForestClassifier是一个基于随机森林算法的分类器，常用于机器学习任务中。

- 随机森林是一种集成学习算法，它将多个决策树组合起来形成一个更强大的分类器。随机森林算法的基本思想是，构建多个决策树，每个树都使用随机选取的特征和样本进行训练，然后将所有树的结果进行投票，选择得票最多的类别作为最终的分类结果。这个过程可以有效地减少随机误差和过拟合。

- 本项目RandomForestClassifier是sklearn库中实现随机森林分类器的类。它提供了许多参数可以调整，以便更好地适应不同的数据集和任务要求，例如：

  - n_estimators：森林中决策树的数量。

  - criterion：用于衡量分裂质量的度量标准，可以是gini或entropy。

  - max_depth：决策树的最大深度。

  - min_samples_split：拆分内部节点所需的最小样本数。

  - min_samples_leaf：在叶节点处所需的最小样本数。

使用RandomForestClassifier可以对数据集进行分类，例如：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成示例数据集
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# 在训练集上拟合模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 输出分类器的准确率
print("Accuracy:", clf.score(X_test, y_test))
```

在上面的示例中，首先使用make_classification函数生成一个示例数据集，然后将数据集分成训练集和测试集。接着创建一个随机森林分类器，并在训练集上拟合模型。最后，在测试集上进行预测，并输出分类器的准确率。

#### GradientBoostingRegressor

- GradientBoostingRegressor是一种基于梯度提升算法的回归模型，常用于机器学习中。

  梯度提升算法是一种集成学习算法，它通过不断迭代来构建多个弱学习器，并将它们组合成一个更强大的学习器。在回归任务中，GradientBoostingRegressor算法的基本思想是，首先用一个简单的回归模型来拟合数据集，然后计算残差，将残差作为新的目标值，再用另一个回归模型来拟合残差，以此类推，直到达到预定的迭代次数或达到指定的误差限制。最终将所有回归模型的结果加权组合起来，形成最终的预测结果。

  GradientBoostingRegressor是sklearn库中实现梯度提升回归的类。它提供了许多参数可以调整，以便更好地适应不同的数据集和任务要求，例如：

  - n_estimators：拟合的树的数量，也就是迭代次数。
  - learning_rate：学习率，控制每个弱学习器对最终结果的贡献大小。
  - max_depth：每个决策树的最大深度，控制模型的复杂度。
  - min_samples_split：每个节点在分裂时需要的最小样本数，用于控制过拟合。
  - loss：损失函数，可以选择平方误差、绝对误差等不同的损失函数。

  在使用GradientBoostingRegressor时，需要根据具体的数据集和任务需求，选择合适的参数来构建模型。同时，还可以通过交叉验证等技术来评估模型的性能和调整参数，以获得更好的预测结果。

#### KNeighborsRegressor

- KNeighborsRegressor是一种基于**K最近邻算法**的回归模型。

  K最近邻算法是一种基于实例的学习方法，它通过比较新样本与训练集中的样本之间的距离来预测新样本的标签。具体而言，对于每个新样本，K最近邻算法会找到它在训练集中K个最近邻居，并将它们的标签的平均值作为新样本的标签。在回归任务中，KNeighborsRegressor算法的基本思想是，根据K个最近邻居的标签的平均值来预测新样本的标签。

  KNeighborsRegressor是sklearn库中实现K最近邻回归的类。它提供了许多参数可以调整，以便更好地适应不同的数据集和任务要求，例如：

  - n_neighbors：所选取的K值，即最近邻居的数量。
  - weights：指定如何计算最近邻居的权重，可以是uniform（所有邻居的权重相等）或distance（邻居的权重与距离成反比）。
  - algorithm：指定用于计算最近邻居的算法，可以是brute（暴力搜索）、kd_tree（基于KD树的搜索）或ball_tree（基于球树的搜索）。
  - metric：指定用于计算距离的度量方法，可以是欧氏距离、曼哈顿距离等不同的度量方法。

  在使用KNeighborsRegressor时，需要根据具体的数据集和任务需求，选择合适的参数来构建模型。同时，还可以通过交叉验证等技术来评估模型的性能和调整参数，以获得更好的预测结果。需要注意的是，KNeighborsRegressor算法对于高维稠密的数据集表现良好，但对于稀疏数据和高维稀疏数据的处理效果可能不好，因此需要根据具体的数据集和任务需求选择合适的算法和度量方法。

- KNN是一种基于实例的机器学习算法，其中的参数包括K值和距离度量方法。在使用KNN算法时，需要注意以下几点：

  1. K值的选择：K值表示在进行分类或回归预测时，考虑的最近邻居的个数。K值的选择对算法性能和预测结果有重要影响。如果K值太小，模型容易受到噪声的影响，导致过拟合；如果K值太大，模型可能会过于简单，导致欠拟合。因此，需要通过交叉验证等方法选择合适的K值。

     - 当K值为偶数时，在进行分类或回归预测时，可能会出现两个或多个最近邻居的标签或值相同的情况，从而无法确定样本的分类或回归值。这种情况下，需要使用一些额外的规则来解决这种歧义。

     - 一种常用的解决方法是使用加权平均法。具体而言，对于K个最近邻居中的每个样本，可以计算它与测试样本之间的距离，并将其作为权重，对它们的标签或值进行加权平均。例如，对于K=4的情况，可以计算测试样本与4个最近邻居之间的距离，并将它们的标签或值分别乘以对应的权重，然后将它们加权平均，以得到测试样本的预测值。

     - 另一种解决方法是随机选择一个标签或值作为最终的预测结果。例如，对于K=4的情况，可以随机选择其中一个最近邻居的标签或值作为预测结果。

     - 需要注意的是，当K值为奇数时，可以避免上述歧义的情况，因为K个最近邻居中必然有一个标签或值出现的次数多于一半，从而可以确定样本的分类或回归值。因此，在实际应用中，通常建议将K值设置为奇数。

  2. 距离度量方法的选择：KNN算法的核心是基于距离度量来计算样本之间的相似度。常用的距离度量方法包括欧式距离、曼哈顿距离、闵可夫斯基距离等。在选择距离度量方法时，需要根据具体的数据特点和任务需求进行选择。

  3. 数据预处理：KNN算法对数据的分布和尺度敏感，因此需要对数据进行预处理，例如标准化、归一化等操作，以使得数据分布更加均匀，尺度更加一致。

  4. 计算效率：KNN算法需要计算每个测试样本与所有训练样本之间的距离，因此当数据集较大时，算法计算复杂度会变得很高。因此，需要考虑使用一些优化技术，例如使用KD树等数据结构，以提高算法的计算效率。

需要注意的是，KNN算法虽然简单易用，但也存在一些局限性，例如对噪声敏感、对维数敏感、计算复杂度高等。因此，在实际应用中需要综合考虑算法的优缺点，选择适合的算法并进行参数调整和优化。

- KNN是K-Nearest Neighbors的缩写，中文翻译为K近邻算法，是一种基于实例的机器学习算法。该算法的核心思想是根据已有的训练样本，通过计算测试样本与训练样本之间的距离，找到与测试样本最近的K个训练样本，然后根据这K个训练样本的标签进行分类或回归。

  KNeighborsRegressor是sklearn库中的一个实现了KNN算法的回归模型。该模型是基于最近邻算法实现的，可以用于连续型数值的预测。KNeighborsRegressor模型接收一个整数K作为参数，该参数表示要考虑的最近邻居的个数。该模型还支持不同的距离度量方法，例如欧式距离、曼哈顿距离等。

  因此，KNeighborsRegressor模型是KNN算法在回归问题上的一种具体实现。和其他机器学习模型一样，KNeighborsRegressor模型也有一些需要注意的参数和超参数，例如K值、距离度量方法、权重函数等，需要根据具体的数据和任务进行设置和调整

#### MLPRegressor

- MLPRegressor是一种基于多层感知器（MLP）神经网络的回归模型，常用于机器学习中。

  多层感知器（MLP）神经网络是一种具有多个隐层的前馈神经网络，它可以用于解决各种分类和回归问题。在回归任务中，MLPRegressor算法的基本思想是，通过多层非线性变换将输入数据映射到一个高维空间中，然后通过输出层将高维空间中的结果映射回原始空间中的标签。具体而言，MLPRegressor算法会在每个隐层中使用多个神经元来学习非线性特征，最终输出一个连续的预测值。

  MLPRegressor是sklearn库中实现MLP神经网络回归的类。它提供了许多参数可以调整，以便更好地适应不同的数据集和任务要求，例如：

  - hidden_layer_sizes：指定每个隐层中的神经元数量和隐层的数量。
  - activation：指定激活函数，可以是sigmoid、ReLU等不同的激活函数。
  - solver：指定用于优化权重的求解器，可以是adam、lbfgs、sgd等不同的求解器。
  - alpha：指定正则化参数，用于控制模型的复杂度。
  - learning_rate：指定学习率的调整策略，可以是constant、invscaling、adaptive等不同的策略。

  在使用MLPRegressor时，需要根据具体的数据集和任务需求，选择合适的参数来构建模型。同时，还可以通过交叉验证等技术来评估模型的性能和调整参数，以获得更好的预测结果。需要注意的是，MLPRegressor算法对于数据集的特征缩放和标准化非常敏感，因此在使用MLPRegressor时需要对数据进行预处理。

#### BaggingClassifier

A Bagging classifier.

A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [[1\]](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#rb1846455d0e5-1). If samples are drawn with replacement, then the method is known as Bagging [[2\]](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#rb1846455d0e5-2). When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [[3\]](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#rb1846455d0e5-3). Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [[4\]](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#rb1846455d0e5-4).

Bagging分类器是一种元估计器集成方法，它在原始数据集的随机子集上分别拟合基分类器，然后聚合它们的个体预测结果（通过投票或平均）以形成最终预测。这种元估计器通常可以用作一种降低黑盒估计器（例如决策树）方差的方法，通过将随机化引入其构建过程，然后将其组合起来。

此算法包括文献中的几项工作。当随机抽取数据集的子集作为样本的随机子集时，此算法称为Pasting [1]。如果使用放回抽样，则该方法称为Bagging [2]。当随机抽取数据集的子集作为特征的随机子集时，该方法称为随机子空间方法[3]。最后，当基估计器基于样本和特征的子集构建时，该方法称为Random Patches [4]。

In ensemble algorithms, bagging methods form a class of algorithms which build several instances of a black-box estimator on random subsets of the original training set and then aggregate their individual predictions to form a final prediction. These methods are used as a way to reduce the variance of a base estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. In many cases, bagging methods constitute a very simple way to improve with respect to a single model, without making it necessary to adapt the underlying base algorithm. As they provide a way to reduce overfitting, bagging methods work best with strong and complex models (e.g., fully developed decision trees), in contrast with boosting methods which usually work best with weak models (e.g., shallow decision trees).

Bagging methods come in many flavours but mostly differ from each other by the way they draw random subsets of the training set:

> - When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [[B1999\]](https://scikit-learn.org/stable/modules/ensemble.html#b1999).
> - When samples are drawn with replacement, then the method is known as Bagging [[B1996\]](https://scikit-learn.org/stable/modules/ensemble.html#b1996).
> - When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [[H1998\]](https://scikit-learn.org/stable/modules/ensemble.html#h1998).
> - Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [[LG2012\]](https://scikit-learn.org/stable/modules/ensemble.html#lg2012).

In scikit-learn, bagging methods are offered as a unified [`BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier) meta-estimator (resp. [`BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor)), taking as input a user-specified estimator along with parameters specifying the strategy to draw random subsets. In particular, `max_samples` and `max_features` control the size of the subsets (in terms of samples and features), while `bootstrap` and `bootstrap_features` control whether samples and features are drawn with or without replacement. When using a subset of the available samples the generalization accuracy can be estimated with the out-of-bag samples by setting `oob_score=True`. As an example, the snippet below illustrates how to instantiate a bagging ensemble of `KNeighborsClassifier` estimators, each built on random subsets of 50% of the samples and 50% of the features.

在集成算法中，bagging方法是一类算法，它在原始训练集的随机子集上构建多个黑盒估计器实例，然后聚合它们的个体预测结果以形成最终预测。这些方法通过将随机化引入其构建过程，并将其组合起来，以降低基本估计器（例如决策树）的方差。在许多情况下，bagging方法是一种非常简单的改进单个模型的方法，而无需调整底层基础算法。由于它们提供了一种减少过拟合的方法，因此bagging方法适用于强大而复杂的模型（例如完全发展的决策树），而提升方法通常适用于弱模型（例如浅层决策树）。

Bagging方法有许多变种，但它们主要通过绘制训练集的随机子集的方式而不同：

当随机抽取数据集的子集作为样本的随机子集时，此算法称为Pasting [B1999]。

如果使用放回抽样，则该方法称为Bagging [B1996]。

当随机抽取数据集的子集作为特征的随机子集时，该方法称为随机子空间方法[ H1998]。

最后，当基估计器基于样本和特征的子集构建时，该方法称为Random Patches [LG2012]。

在scikit-learn中，Bagging方法提供了一个统一的BaggingClassifier元估计器（resp. BaggingRegressor），以用户指定的估计器为输入，并指定绘制随机子集的策略的参数。特别地，max_samples和max_features控制子集的大小（以样本和特征为单位），而bootstrap和bootstrap_features控制是否使用有放回或无放回的方式来抽取样本和特征。当使用可用样本的子集时，可以通过设置oob_score=True来估计外部样本的泛化精度。例如，下面的代码段说明如何实例化一个KNeighborsClassifier估计器的bagging集合，每个估计器都是在50％的样本和50％的特征的随机子集上构建的。

#### BaggingRegressor

BaggingRegressor是一种基于袋装法（Bagging）的回归模型，常用于机器学习中。

袋装法是一种集成学习方法，它通过随机抽样生成多个训练数据集，并使用这些训练数据集来构建多个基学习器。在回归任务中，BaggingRegressor算法的基本思想是，将训练数据随机分成多个子集，然后使用每个子集来训练一个独立的回归模型，并将所有模型的结果进行平均或加权平均。

BaggingRegressor是sklearn库中实现袋装法回归的类。它提供了许多参数可以调整，以便更好地适应不同的数据集和任务要求，例如：

- base_estimator：指定基学习器的类型，可以是任意的回归模型。
- n_estimators：指定袋装法中的基学习器数量，也就是子模型的数量。
- max_samples：指定每个子集中的样本数量，用于控制过拟合。
- max_features：指定每个子集中的特征数量，用于控制模型的复杂度。
- bootstrap：指定是否进行有放回的随机抽样。

在使用BaggingRegressor时，需要根据具体的数据集和任务需求，选择合适的参数来构建模型。同时，还可以通过交叉验证等技术来评估模型的性能和调整参数，以获得更好的预测结果。需要注意的是，BaggingRegressor算法在处理高偏差低方差的模型时效果比较好，例如决策树等模型。

### 其他

#### 一个基础的分类教程

- [How to Make a Speech Emotion Recognizer Using Python And Scikit-learn - Python Code (thepythoncode.com)](https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn)

#### 相关api

#### skearn.ensemble

- sklearn.ensemble是sklearn库中的一个模块，用于实现集成学习相关的算法。集成学习是一种通过结合多个基学习器来构建一个更强大的学习器的方法，常用于解决各种分类和回归问题。

  sklearn.ensemble模块中包含了许多经典的集成学习算法，例如：

  - Bagging：基于袋装法的集成学习算法，用于构建多个独立的基学习器。
  - RandomForest：基于随机森林的集成学习算法，也是一种基于决策树的集成方法。
  - AdaBoost：基于自适应增强的集成学习算法，用于加权组合多个基学习器。
  - GradientBoosting：基于梯度提升的集成学习算法，用于逐步提升模型的预测能力。
  - Voting：基于投票法的集成学习算法，用于组合多个不同类型的基学习器。

  除了以上常见的集成学习算法外，sklearn.ensemble模块还包含了一些其他的集成学习算法，例如ExtraTreesRegressor和ExtraTreesClassifier等。

  sklearn.ensemble模块提供了统一的API接口，方便用户调用不同的集成学习算法。同时，它也提供了许多参数可以调整，以便更好地适应不同的数据集和任务要求。例如，可以通过调整n_estimators、max_depth、min_samples_split等参数来控制模型的复杂度和泛化能力。

  在使用sklearn.ensemble模块时，需要根据具体的数据集和任务需求，选择合适的算法和参数来构建模型。同时，还可以通过交叉验证等技术来评估模型的性能和调整参数，以获得更好的预测结果。需要注意的是，集成学习算法在处理高方差低偏差的模型时效果比较好，例如决策树等模型。

#### SVR

SVR（Support Vector Regression）是一种基于支持向量机（SVM）的回归模型

支持向量机是一种二分类模型，它通过寻找一个最优超平面来将数据分成两类。在回归任务中，SVR算法的基本思想是，通过寻找一个最优的超平面来拟合数据，并将预测结果限制在一定的范围内。具体而言，SVR算法会将训练数据映射到高维空间中，然后在高维空间中寻找一个最优的超平面，使得所有样本点到超平面的距离都小于一个给定的阈值。最终，SVR算法会输出一个连续的预测值，可以通过调整阈值来控制预测的精度和泛化能力。

- SVR是sklearn库中实现SVM回归的类。它提供了许多参数可以调整，以便更好地适应不同的数据集和任务要求，例如：

  - kernel：指定核函数的类型，常用的有线性核、多项式核和径向基函数（RBF）核等。

  - C：指定正则化参数，用于控制模型的复杂度和泛化能力。

  - epsilon：指定预测结果的精度范围，用于控制模型的鲁棒性和稳定性。

- 在使用SVR时，需要根据具体的数据集和任务需求，选择合适的参数来构建模型。同时，还可以通过交叉验证等技术来评估模型的性能和调整参数，以获得更好的预测结果。需要注意的是，SVR算法在处理高维度、少样本、非线性问题时效果比较好，但在数据量较大时可能会耗费大量的计算资源。

- SVC用于处理分类问题，而SVR用于处理回归问题。但它们的核心思想和算法是相似的，都是通过寻找最优的超平面来拟合数据。同时，它们都可以调整正则化参数和核函数等参数来控制模型的复杂度和泛化能力。

#### 用回归器解决分类问题

虽然回归模型通常用于解决连续变量的预测问题，但在某些情况下可以将回归模型用于解决分类问题。具体而言，可以将回归模型的输出转换为分类标签，从而实现分类任务。

其中一种常见的方法是使用阈值来将回归输出二值化。具体而言，可以选择一个阈值，将回归输出大于该阈值的样本标记为正类，小于该阈值的样本标记为负类。这种方法常用于解决二分类问题。例如，在线性回归模型中，可以选择0.5作为阈值，将输出大于0.5的样本标记为正类，小于0.5的样本标记为负类。

另一种常见的方法是使用逻辑函数（如sigmoid函数）将回归输出映射到[0,1]区间上，并将映射后的输出视为正类概率。

## scikit-learn 加速

    Windows 64-bit packages of scikit-learn can be accelerated using scikit-learn-intelex.
    More details are available here: https://intel.github.io/scikit-learn-intelex
    
    For example:
    
        $ conda install scikit-learn-intelex
        $ python -m sklearnex my_application.py

## DeepLearning method

### Tensorflow.Keras

- Keras是一个用于构建和训练深度学习模型的高级神经网络API，由Francois Chollet创建。以下是Keras的主要信息：

  - Keras支持多种深度学习模型，包括卷积神经网络(CNN)、循环神经网络(RNN)、自编码器(Autoencoder)等。
  - Keras提供了一个简单的API，使得构建和训练深度学习模型变得更加容易和快速。
  - Keras的后端引擎支持多种深度学习框架，包括TensorFlow、Theano和CNTK等，因此可以在不同的硬件和软件平台上运行。
  - Keras的文档和社区支持非常完善，有大量的教程、示例和论坛可以供用户参考和交流。
  - Keras在2017年正式被纳入TensorFlow的官方API，成为了TensorFlow 2.0的默认API。

  总的来说，Keras是一个功能强大、易于使用、高度灵活的深度学习API，为用户提供了一个快速构建和训练深度学习模型的平台。

#### 主要内容

- Keras主要内容包括以下方面：
  1. 模型构建：Keras提供了简单的API，使得用户可以轻松地构建各种深度学习模型，包括卷积神经网络(CNN)、循环神经网络(RNN)、自编码器(Autoencoder)等。
  2. 模型训练：Keras可以使用多种优化器和损失函数来训练模型，并支持多种训练技巧，如批量归一化、dropout等，以提高模型的性能和鲁棒性。
  3. 模型评估：Keras提供了多种评估指标，如准确率、精度、召回率等，可以帮助用户评估模型的性能。
  4. 模型部署：Keras支持多种后端引擎，包括TensorFlow、Theano和CNTK等，因此可以在不同的硬件和软件平台上运行，并支持多种部署方式，如本地部署、云端部署等。
  5. 模型调优：Keras支持网格搜索和随机搜索等方式来进行模型调优，并提供了可视化工具，帮助用户分析模型在训练中的表现。
  6. 社区支持：Keras拥有一个庞大的社区，在网站上提供了大量的文档、教程、示例和论坛等资源，用户可以方便地获取帮助和交流。

- [KerasBasic](KerasBasic)

### RNN+LSTM

- RNN（Recurrent Neural Network，循环神经网络）是一类用于处理序列数据的神经网络，它的每个时间步都会接收一个输入和一个来自上一个时间步的隐藏状态，并输出一个新的隐藏状态和一个输出。

- LSTM（Long Short-Term Memory，长短时记忆网络）和GRU（Gated Recurrent Unit，门控循环单元）都是RNN的变体，旨在解决RNN的梯度消失和梯度爆炸问题，以及长期依赖性问题。LSTM和GRU都是通过引入门机制来控制信息的流动，从而使得模型可以更好地记忆长期依赖性信息。LSTM引入了三个门（输入门、遗忘门和输出门），GRU则引入了两个门（重置门和更新门）。这些门控制了状态的更新，使得模型可以更好地捕捉序列中的关键信息。因此，LSTM和GRU相对于传统的RNN具有更好的性能。

- 总之，LSTM和GRU是RNN的改进版本，它们在处理序列数据时可以更好地捕捉长期依赖性信息。


## 优化

### dropout@正则化技术@泛化能力

- Dropout是一种用于深度神经网络的正则化技术，旨在减少过拟合（overfitting）的发生。
- Dropout在训练过程中**随机地**将一些神经元的输出设置为0，从而强制使神经网络中的**每个神经元**都变得不可或缺，因此可以更好地泛化到新数据上。
- 具体来说，Dropout的操作是在**每个训练批次**中，随机选择一些**神经元**，并将它们的输出设置为0。这些被选择的神经元在<u>该批次中将不会收到反向传播的梯度更新</u>。这样一来，每个神经元都必须学会与其他神经元合作来完成任务，从而使得神经网络具有更好的泛化能力。
- Dropout通常在深度神经网络的全连接层或卷积层中使用。
- 在实践中，Dropout的使用可以通过在模型中添加**Dropout层**来实现，例如在Keras中，可以使用`keras.layers.Dropout()`函数来添加Dropout层。
- 需Dropout只应该在训练过程中使用，而不应该在测试过程中使用。在测试过程中，应该使用所有的神经元来进行预测，以获得更准确的结果。
- 因此，在测试过程中，可以通过在训练过程中使用Dropout时，对每个神经元的输出进行缩放来实现。这种缩放可以通过在Keras中使用`model.predict()`函数的`predict()`方法来实现。

### 数据平衡balance@shuffle🎈

#### balance

- 机器学习的许多接口的参数中提供了`balance` 参数,一般是一个布尔值，用于控制是否在训练和测试数据集中进行数据平衡。
- 如果将 `balance` 设置为 `True`，则在划分训练和测试数据集之前，会对原始数据集进行重新采样，以使得每个类别的样本数量相等或接近相等。
- 这可以避免训练和测试数据集中类别不平衡导致的模型偏差和泛化性能下降。
- 注意,由于balance操作对于数据集划分有一定要求,不是任何数据集都可以执行balance操作(例如三分类中,test set中只缺失了某一个类别的样本,这中情况下执行balance,将导致测试集样本数量为空)
- 因此,即使我们对ER对象传入balance=True,依然不能保证balance能够执行,调用load_data后,需要检查返回的balance操作是否执行的结果

#### shuffle

- 通常,只要数据集结构合理,那么利用shuffle操作(随机打乱样本顺序),可以有效减少balance失败的情况
- 往往数据集都是有序的,这就容易导致样本不均衡导致的模型泛化能力下降等不良影响
- 在本项目中,shuffle被安排在`audio.create_meta`模块中实现,也就是创建语音文件元数据的地方,

##  识别系统的模块和结构🎈

- 另见文档:[ProjectStructure.md](ProjectStructure)

## 客户端开发文档

- 另见文档:[ccser_client.md](ccser_client.md)

## 开发工具

- IDEA
  - python plugin
  - pylint 
- vscode
  - python extension
  - codeium AI extension
  - AWS CodeWhisper extension
  - CodeGeex extension
  - ...
- 代码阅读
  - 使用大纲阅读一个大文件是好办法
  - 对于没有封装在函数或者类中的代码,可以设置`fold level`折叠至level 2来快速把握代码结构

# 部分实验结果

## 同库识别

- 同库识别相对简单,识别率还可以,自行实验

## 跨库识别

- 许多实验记录保存在`recognizer\cross_SER_result`目录中(相对有项目的根目录的路径)
- 这些结果放在各自的`ipynb`文件中,代码部分一并保留,如果只想查看实验结果,可以折叠代码



### angry&sad@emodb-ravdess

- `train_emodb_AS.csv+test_ravdess_AS.csv`

  - ```bash
    meta_files\train_emodb_AS.csv meta_files\test_ravdess_AS.csv
    meta_files\test_ravdess_AS.csv @{test_meta_files} in load_data_from_meta
    partition='train'
    meta_files\train_emodb_AS.csv @{meta_files}in load_data_preprosscing
    cast the 'meta_files\train_emodb_AS.csv' to [str]
    meta_files\train_emodb_AS.csv @🎈{meta_file}
    [I] Loading audio file paths and its corresponding labels...
    meta_files: ['meta_files\\train_emodb_AS.csv']
    存在meta_files\train_emodb_AS.csv文件!
    meta_path='meta_files\\train_emodb_AS.csv'@
    db='emodb'@
    检查特征文件features\emodb_mfcc_AS_151.npy是否存在...
    self.e_config=['angry', 'sad']
    特征矩阵文件(.npy)已经存在,直接导入:loading...
    (151, 40) @{feature.shape}
    [Info] Adding  train samples
    partition='test'
    meta_files\test_ravdess_AS.csv @{meta_files}in load_data_preprosscing
    cast the 'meta_files\test_ravdess_AS.csv' to [str]
    meta_files\test_ravdess_AS.csv @🎈{meta_file}
    [I] Loading audio file paths and its corresponding labels...
    meta_files: ['meta_files\\test_ravdess_AS.csv']
    存在meta_files\test_ravdess_AS.csv文件!
    meta_path='meta_files\\test_ravdess_AS.csv'@
    db='ravdess'@
    检查特征文件features\ravdess_mfcc_AS_406.npy是否存在...
    self.e_config=['angry', 'sad']
    特征矩阵文件(.npy)已经存在,直接导入:loading...
    (406, 40) @{feature.shape}
    [Info] Adding  test samples
    [+] Data loaded
    2796338002912
    None @{self.model}
    Evaluating RandomForestClassifier:   0%|          | 0/5 [00:00<?, ?it/s]
    SVC(C=10, gamma=0.001) @{self.model}
    RandomForestClassifier(max_depth=7, max_features=0.5, n_estimators=40) @{self.model}
    Evaluating KNeighborsClassifier:  40%|████      | 2/5 [00:00<00:00, 14.40it/s]  
    KNeighborsClassifier(n_neighbors=3, p=1, weights='distance') @{self.model}
    Evaluating MLPClassifier:  40%|████      | 2/5 [00:00<00:00, 14.40it/s]       d:\condaPythonEnvs\tf2.10\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:603: UserWarning: Got `batch_size` less than 1 or larger than sample size. It is going to be clipped
      warnings.warn(
    Evaluating BaggingClassifier:  80%|████████  | 4/5 [00:00<00:00,  6.96it/s]
    MLPClassifier(alpha=0.01, batch_size=512, hidden_layer_sizes=(300,),
                  learning_rate='adaptive', max_iter=400) @{self.model}
    BaggingClassifier(max_features=0.5, n_estimators=50) @{self.model}
    Evaluating BaggingClassifier: 100%|██████████| 5/5 [00:00<00:00,  7.09it/s]
    [+] Best model : RandomForestClassifier with 77.094% test accuracy
    test_score=0.770935960591133
    ```

### ravdess-emodb

```bash
(d:\condaPythonEnvs\tf2.10) PS D:\repos\CCSER\SER> py "d:\repos\CCSER\SER\recognizer\basic.py"
@{model}
partition='train'
D:\repos\CCSER\SER\meta_files\train_ravdess_AS.csv @🎈{meta_file}
[I] Loading audio file paths and its corresponding labels...
meta_file存在D:\repos\CCSER\SER\meta_files\train_ravdess_AS.csv文件!
检查特征文件D:\repos\CCSER\SER\features\ravdess_mfcc_AS_1216.npy是否存在...
self.e_config=['angry', 'sad']
self.f_config=['mfcc']
npy文件不存在,尝试创建...
{} @{self.feature_transforms}🎈
True @{save_obj}
Extracting features for partition:: 100%|███████████████████████████████████████████████████████████| 1216/1216 [00:19<00:00, 63.43it/s] 
fts参数key合法
🎈🎈🎈特征提取
(1216, 40) @{feature.shape}
[Info] Adding  train samples
partition='test'
D:\repos\CCSER\SER\meta_files\test_emodb_AS.csv @🎈{meta_file}
[I] Loading audio file paths and its corresponding labels...
meta_file存在D:\repos\CCSER\SER\meta_files\test_emodb_AS.csv文件!
检查特征文件D:\repos\CCSER\SER\features\emodb_mfcc_AS_38.npy是否存在...
self.e_config=['angry', 'sad']
self.f_config=['mfcc']
特征矩阵文件(.npy)已经存在,直接导入:loading...
(38, 40) @{feature.shape}
[Info] Adding  test samples
[I] Data loaded

self.ae=<audio.extractor.AudioExtractor object at 0x0000028D91CE85E0>
self.ae.pca=None🎈
Evaluating <SVC>:   0%|                                                                                           | 0/5 [00:00<?, ?it/s]@{model}

[I] SVC with 0.7105263157894737 test accuracy
Evaluating <RandomForestClassifier>:  20%|████████████▊                                                   | 1/5 [00:05<00:23,  5.80s/it]@{model}
Evaluating <BaggingClassifier>: 100%|█████████████████████████████████████████████████████████████████████| 5/5 [00:14<00:00,  2.99s/it] 
[🎈] Best model : RandomForestClassifier with 97.368% test accuracy
train_score=0.9868421052631579verbose=0               precision    recall  f1-score   support

       angry       1.00      0.92      0.96        24
         sad       0.88      1.00      0.93        14

    accuracy                           0.95        38
   macro avg       0.94      0.96      0.94        38
weighted avg       0.95      0.95      0.95        38 RandomForestClassifier
test_score=0.9473684210526315
(1216, 40) (1216,) 🎈
n_splits=5
cv_score=0.9508196721311475
(d:\condaPythonEnvs\tf2.10) PS D:\repos\CCSER\SER>
```





