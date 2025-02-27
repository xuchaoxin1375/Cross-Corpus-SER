
##
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from config.MetaPath import get_example_audio_file
audio_file = get_example_audio_file()

def showWaveForm(audio_file=audio_file):
    # 加载音频文件
    y, sr = librosa.load(audio_file, sr=None)

    # 绘制波形图
    figsize=(10, 5)
    plt.figure(figsize=figsize)
    librosa.display.waveshow(y, sr=sr)

    plt.title(f'Waveform @ {audio_file}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()

##
# 频谱图


# 加载音频文件
def showFreqGraph(audio_file=audio_file):
    y, sr = librosa.load(audio_file)

    # 计算短时傅里叶变换
    D = librosa.stft(y)

    # 绘制频谱图
    # plt.figure(figsize=(10, 4))

    librosa.display.specshow(librosa.amplitude_to_db(abs(D), ref=np.max), y_axis='linear', x_axis='time')

    plt.colorbar(format='%+2.0f dB')

    # plt.title('线性频谱图')
    # plt.xlabel('时间')
    # plt.ylabel('频率')
    plt.show()


def showMelFreqGraph(audio_path=audio_file):
    """
    Generate and display a Mel-frequency spectrogram from an audio file.

    Args:
        audio_path (str): Path of the audio file to load.

    Returns:
        None
    """
    y, sr = librosa.load(audio_path)

    # 计算Mel频谱
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    #另一种等效写法:(使用S参数)
    # D=np.abs(librosa.stft(y))**2
    # S=librosa.feature.melspectrogram(S=D, sr=sr, n_mels=128) 

    # 将Mel频谱转换为分贝表示
    S_db = librosa.power_to_db(S, ref=np.max)

    # 绘制Mel频谱(系数)图
    fig,ax=plt.subplots()
    # plt.figure(figsize=(10, 4))
    img=librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000,ax=ax)

    fig.colorbar(img,ax=ax,format='%+2.0f dB')
    ax.set(title='Mel-frequencey spectrogram')

    # plt.title('Mel频谱图')
    # plt.xlabel('时间')
    # plt.ylabel('频率（Mel刻度）')
    plt.show()
def showMFCCGraph(audio_path=audio_file):
    pass

if __name__=="__main__":
    pass
    # showWaveForm()
    # showFreqGraph()
    showMelFreqGraph()
##