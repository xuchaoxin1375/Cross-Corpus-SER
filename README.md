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

- 语音识别和语音情感识别的主要区别在于关注点不同。语音识别关注语言内容，而语音情感识别关注情感状态。
- 语音情感识别和跨语料库语音情感识别的主要区别在于应用范围。语音情感识别通常针对单一语言或文化背景，而跨语料库语音情感识别需要处理多种语言和文化背景下的情感识别问题。并且通常是在单个库（而非混合库）上训练识别模型，在其他库上进行语音情感识别任务。

# Deploy

## Requirements

- 在conda环境中(推荐使用全新创建的环境)
  - 根据我的经验来看,我们只用conda创建环境,以及按照python
  - 但是安装第三方包的时候 `conda install`往往不是一个好的选择
  - 如果不是包官方推荐,那么建议都用 `pip install`安装
    - 例如pytorch支持用conda安装
    - tensorflow推荐用pip 安装
  - 特别是当包的依赖关系复杂,每次安装要分析很久,当项目依赖的包版本都清楚可用,那么意味着检查依赖这一步不需要了,使用 `pip install`会高效的多
- 安装python `conda install python=3.9`
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
- 创建环境的时候注意带上python,或者创建一个空环境后,先使用 `conda install python=3.9`来安装python,否则后续的 `pip install`将无法正常工作
- python的版本受到许多依赖的限制,尤其是深度学习框架(比如 `tensorflow`),初次之外,python的版本可以根据需要调整,为了兼容性起见,通常推荐python的版本举例最新发布的python版本低2到3个中版本,例如当前的最新版本是 `3.m.n`那么推荐那安装 `3.(m-2)`,最后的n通常无关紧要,在这种情况下一些第三方包一般都适配到了 `3.(m-n)`

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

- 我在一次调试matplotlib backend设置的过程中,偶然发现在notebook中使用 `%matplotlib qt`失败
- 后来我尝试卸载重装matplotlib和pyside
- 然而提示我一些 `dll`问题和权限错误问题
- 这些问题时平时不曾遇到
- 我猜测可能时机器太久没有关机了(平时我都是休眠),导致系统出现了一些错误

  - 机器发生错误是很有可能的,就比如学校图书馆的刷脸系统,`验证成功`平时的话屏幕显示绿色提醒,然而最近 `验证成功`显示的也是红色
  - 而在早期的windows7上,有时候从休眠中回复直接会失败
- 然后我重启机器,发现任务栏多出了个搜索框,系统更新了没有重启可能也造成了一些影响
- 有时候还会出现各种意外的问题

  - 例如 `KNeighborsClassifier`训练完毕后,用它预测新的样本或在训练集上预测会报错:

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

- 可以通过设置 `desc`参数来为进度条添加描述信息：

```python
from tqdm import tqdm
import time

for i in tqdm(range(10), desc="Processing data"):
    time.sleep(0.1)
```

这将在进度条上方添加一个描述信息，描述当前正在执行的任务。

- 可以通过设置 `leave`参数来控制进度条是否应该在循环结束后保留：

```python
from tqdm import tqdm
import time

for i in tqdm(range(10), leave=False):
    time.sleep(0.1)
```

这将在循环结束后清除进度条。

这些是 `tqdm`的一些常见用法，它还具有许多其他有用的功能，如支持多线程和多进程，可以自定义进度条的样式等。

- 本项目中遍历 `estimators`
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

- 为了放便地运行本项目,请配置环境变量 `PYTHONPATH`,并至少向其中添加本项目的根目录
- 例如,我的项目放在 `$ccser=D:\repos\CCSER\SER`,根据自己的clone/download位置情况修改这个值
- 例如,在使用powershell配置(windows系统下)

  - ```powershell
    $ccser="D:\repos\CCSER\SER"
    setx PYTHONPATH "$ccser;$env:PYTHONPATH"
    ```
  - 执行:

    - ```powershell
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

### SAVEE

#### Speakers

'DC', 'JE', 'JK' and 'KL' are four male speakers recorded for the SAVEE database

---

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
  | W      | anger             | Ärger (Wut)     | A                   |
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
- 不过由于语料库的命名规范的不同,您或许要亲自编写针对于db的 `create_{db}_meta()`函数,

## 语料库文件在项目中的组织与划分

- 将EMODB语料库放在单独的目录 `data/emodb`
- 将RAVDESS语料库

  - 大部分样本放在了训练集目录 `data/ravdess`
- 测试集(验证集)目录 `data/validation`

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

### 统计项目大小

```bash
# cxxu_u22 @ cxxuWn11 in /mnt/d/repos/CCSER/SER on git:main x [10:21:23]
$ du -h --max-depth=1 |sort -hr
1.8G    .
1.6G    ./data
96M     ./.git
19M     ./SG
18M     ./.vscode
16M     ./features
14M     ./charts
11M     ./test_playground
4.6M    ./recognizer
1.2M    ./assets
1.1M    ./meta_files
932K    ./audio
92K     ./__pycache__
76K     ./config
32K     ./grid
16K     ./archive
16K     ./.idea
8.0K    ./recognize_result
```

### meta_files(csv 元数据文件)🎈

- 由于不同语料库的文件名规范不同,所以在使用前应该进行基本的统一处理(主要抽取语音文件路径和文件的情感标签)
- 可以利用pandas将抽取结果持久化保存到磁盘,以便后续使用

  - 特别是对于大量文件来说,保存处理结果有利于提高重复试验的效率
- desc_files (csv)文件存储的是语料库的语音文件的路径(path),情感类别(emotion),存放在项目根目录的 `meta_files`目录下

### Emotions available@情感类别

- 本项目采用的语料库中包含 "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "surprise","ps" (pleasant surprise) and "boredom".
- 尽管如此,您不总是可以任意搭配情感组合,因为有些语料库缺失其中的若干种情感
- 比较常见的组合包括:
  - angry,happy,sad
  - angry,happy,sad,neutral,pleasant

## Feature Extraction@特征提取🎈

- 特征提取是语音情感识别系统的主要部分。它基本上是通过将语音波形转换为参数形式的表示形式，以相对较低的数据速率完成的。
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
- 他们的名称结构是 `{partition}_{feature_label}_{first_letters}_{n_samples}.npy`
  - partition$\in${'test','train'}
  - feature_label表示提取了哪些特征(特征链,用 `-`连接不同特征)
  - first_letters表示情感种类(单词的首字母)
    - 为了控制数据大小,预设2种情感模式(HNS和AHNPS)

## Grid Search@网格搜索

- [3.2. Tuning the hyper-parameters of an estimator — scikit-learn documentation](https://scikit-learn.org/stable/modules/grid_search.html)
- 网格搜索是一种机器学习中的超参数调优技术，其目的是找到模型超参数的最优值。
- 超参数是在训练过程中不会被学习的参数，但在训练前需要设置，可以对模型性能产生重要影响。
- Grid Search 可以翻译为“网格搜索”或“网格调参”，是一种常用的机器学习超参数调优方法。
- Grid Search 的目的是通过遍历给定的超参数组合，找到最优的超参数组合，以获得最佳的模型性能。它基于一个预定义的超参数网格（grid），对每个超参数组合进行评估和比较，从而选择最佳的超参数组合。
- 网格搜索涉及 `<u>`定义一个超参数值的矩阵`</u>`，并且系统地搜索矩阵以寻找在验证数据集上表现最佳的超参数组合。
- 具体来说，Grid Search 将每个超参数的取值范围划分成一组离散的值，然后对所有可能的超参数组合进行遍历，对每个组合训练一个模型，并使用交叉验证等方法评估模型性能。
- 最后，选择具有最佳性能的超参数组合作为最终模型的超参数。
- Grid Search 是一种简单而有效的调参方法，但它需要遍历所有可能的超参数组合，因此计算成本较高。为了减少计算成本，可以使用 `<u>`随机搜索（Random Search）`</u>`等其他调参方法。

### 计算best_model

- `grid.search`模块提供了计算各个算法最优超参数的实现

# 本项目使用的算法(分类模型)🎈

## ML Classifiers

- SVC
- RandomForestClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- DecisionTreeClassifer
- MLPClassifier
- BaggingClassifier
- AdaBoostingClassifer
- StackingClassifier
- Recurrent Neural Networks (Keras)

## 其他

### 一个基础的分类教程

- [How to Make a Speech Emotion Recognizer Using Python And Scikit-learn - Python Code (thepythoncode.com)](https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn)

### 相关api

### skearn.ensemble

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



## 模型评估方案(交叉验证法评估)

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

### 使用shuffle

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
- 具体来说，Dropout的操作是在**每个训练批次**中，随机选择一些**神经元**，并将它们的输出设置为0。这些被选择的神经元在`<u>`该批次中将不会收到反向传播的梯度更新`</u>`。这样一来，每个神经元都必须学会与其他神经元合作来完成任务，从而使得神经网络具有更好的泛化能力。
- Dropout通常在深度神经网络的全连接层或卷积层中使用。
- 在实践中，Dropout的使用可以通过在模型中添加**Dropout层**来实现，例如在Keras中，可以使用 `keras.layers.Dropout()`函数来添加Dropout层。
- 需Dropout只应该在训练过程中使用，而不应该在测试过程中使用。在测试过程中，应该使用所有的神经元来进行预测，以获得更准确的结果。
- 因此，在测试过程中，可以通过在训练过程中使用Dropout时，对每个神经元的输出进行缩放来实现。这种缩放可以通过在Keras中使用 `model.predict()`函数的 `predict()`方法来实现。

### 数据平衡balance@shuffle🎈

#### balance

- 机器学习的许多接口的参数中提供了 `balance` 参数,一般是一个布尔值，用于控制是否在训练和测试数据集中进行数据平衡。
- 如果将 `balance` 设置为 `True`，则在划分训练和测试数据集之前，会对原始数据集进行重新采样，以使得每个类别的样本数量相等或接近相等。
- 这可以避免训练和测试数据集中类别不平衡导致的模型偏差和泛化性能下降。
- 注意,由于balance操作对于数据集划分有一定要求,不是任何数据集都可以执行balance操作(例如三分类中,test set中只缺失了某一个类别的样本,这中情况下执行balance,将导致测试集样本数量为空)
- 因此,即使我们对ER对象传入balance=True,依然不能保证balance能够执行,调用load_data后,需要检查返回的balance操作是否执行的结果

#### shuffle

- 通常,只要数据集结构合理,那么利用shuffle操作(随机打乱样本顺序),可以有效减少balance失败的情况
- 往往数据集都是有序的,这就容易导致样本不均衡导致的模型泛化能力下降等不良影响
- 在本项目中,shuffle被安排在 `audio.create_meta`模块中实现,也就是创建语音文件元数据的地方,

# 识别系统的模块和结构🎈

- 另见文档:[ProjectStructure.md](ProjectStructure)

# 客户端开发文档

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
  - 对于没有封装在函数或者类中的代码,可以设置 `fold level`折叠至level 2来快速把握代码结构

# 部分实验结果

## 同库识别

- 同库识别相对简单,识别率还可以,自行实验

## 跨库识别

- 许多实验记录保存在 `recognizer\cross_SER_result`目录中(相对有项目的根目录的路径)
- 这些结果放在各自的 `ipynb`文件中,代码部分一并保留,如果只想查看实验结果,可以折叠代码

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
