

"""
A utility script used for converting audio samples to be 
suitable for feature extraction
"""

import os

def convert_audio(audio_path, target_path, remove=False):
    """This function sets the audio `audio_path` to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system.
        Return:if covert successfully,return 0;else return non-zero value
        """
    # 要求系统有可运行的ffmpeg,这里命令行的方式执行转换
    #调用本函数时应该确保target_path存在(否则应该提前建立,然后调用本函数)
    v = os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")

    #ffmpeg usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...
    # -ac channels        set number of audio channels
    # -ar rate            set audio sampling rate (in Hz)
    # os.system(f"ffmpeg -i {audio_path} -ac 1 {target_path}")
    if remove:
        os.remove(audio_path)
    return v #如果命令行执行失败,返回一个非0值(表示出错)


def convert_audios(dir_path, target_path, remove=False):
    """Converts a path of wav files to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
        and then put them into a new folder called `target_path`
            Params:
                dir_path (str): the dir_path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system.
        convert_audio的批处理版本,通过调用convert_audio实现
        """

    for dirpath, dirnames, filenames in os.walk(dir_path):
        # 每个"子目录"在建立一个循环
        for dirname in dirnames:
            # 构造dir_path的子目录的绝对路径
            dirname = os.path.join(dirpath, dirname)
            # 计算目标目录
            target_dir = dirname.replace(dir_path, target_path)
            # target_dir=os.path.join(target_path,dirname)
            # 目标目录缺失时立即创建
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            if file.endswith(".wav"):
                # it is a wav file
                target_file = file.replace(dir_path, target_path)
                convert_audio(file, target_file, remove=remove)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Convert ( compress ) wav files to 16MHz and mono audio channel ( 1 channel )
                                                    This utility helps for compressing wav files for training and testing""")
    parser.add_argument("audio_path", help="Folder that contains wav files you want to convert")
    parser.add_argument("target_path", help="Folder to save new wav files")
    parser.add_argument("-r", "--remove", type=bool, help="Whether to remove the old wav file after converting", default=False)

    args = parser.parse_args()
    audio_path = args.audio_path
    target_path = args.target_path

    if os.path.isdir(audio_path):
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
            convert_audios(audio_path, target_path, remove=args.remove)
    elif os.path.isfile(audio_path) and audio_path.endswith(".wav"):
        if not target_path.endswith(".wav"):
            target_path += ".wav"
        convert_audio(audio_path, target_path, remove=args.remove)
    else:
        raise TypeError("The audio_path file you specified isn't appropriate for this operation")
