"""
EF= Emotions Features
配置项目中常用到的特征字组合和情感组合
主要的形式为列表和字典
特征组合配置有时也表示为audio_config
"""
AVAILABLE_EMOTIONS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "ps", # pleasant surprised
    "boredom",
    "others"
}
#deprecated
def get_f_config_dict(features_list)->dict[str,bool]:
    """
    解析features_list中的特征,并检查特征要求是否在识别计划内
    Converts a list of features into a dictionary understandable by
    `data_extractor.AudioExtractor` class
    返回值:一个bool字典,对各种特征的开关
    """
    #计划最多提取5种,默认不提取任何一种
    # f_config_default = {'mfcc': False, 'chroma': False, 'mel': False, 'contrast': False, 'tonnetz': False}
    f_config={}
    for feature in features_list:
        validate_feature(feature)
        # 将提出的合法特征的提取请求修改为True
        f_config[feature] = True
    return f_config

def validate_feature(feature):
    """验证特征feature是否是合法的(单个特征验证)

    Parameters
    ----------
    feature : str
        被验证的情感特征字符串

    Raises
    ------
    TypeError
        特征字符串不可识别
    """
    if feature not in ava_features:
        raise TypeError(f"Feature passed: {feature} is not recognized.only features in {ava_features} are supported!")

def validate_emotions(emotions,Noneable=False):
    """验证emotions参数是否都是有效的情感标签
    注意这里也排查emotions是否为空的情况

    params
    -
    emotions:list[str]
    
    Note
    - 
    粗糙的实现
    if(set(emotions)<=set(ava_emotions)):
        return True
    else:
        type_error = TypeError("Invalid type of emotions!")
        raise type_error

    Parameters
    ----------
    emotions : list[str]
        判断其中的情感字符串都是受支持的情感
    """
    if emotions is None and Noneable == False:
        raise TypeError("Emotions is None!🎈")
    # 提供对单个情感字符串的支持(包装为list)
    if isinstance(emotions,str):
        emotions=[emotions]
    # print(emotions)
    for e in emotions:
        if e not in ava_emotions:
            raise TypeError(f"Emotion passed: {e} is not recognized.")
    return emotions

# 预设特征组合
ava_features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
mfcc,chroma,mel,contrast,tonnetz=ava_features
ava_emotions=sorted(list(AVAILABLE_EMOTIONS))
# #最基础的3情感组合
MCM = ["mfcc", "chroma", "mel"]
MCM_dict = get_f_config_dict(MCM)
# 预设情感组合
HNS = ["sad", "neutral", "happy"]
AHNPS = ["sad", "neutral", "happy", "ps", "angry"]
emotions_extend_dict={e[0].upper():e for e in ava_emotions}
HNS_dict = {"sad": 1, "neutral": 2, "happy": 3}
AHNPS_dict = {"angry": 1, "sad": 2, "neutral": 3, "ps": 4, "happy": 5}
# 语料库的标签对应关系
# #解析emodb语料库的语音文件名和情感标签对应关系
categories_emodb = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral"
}
# The letters 'a', 'd', 'f', 'h', 'n', 'sa' and 'su' represent 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise' emotion classes respectively. 
#为了和其他库标签接轨,这里需要将名词转换为形容词(也可以将其他库的形若词转为名词)
categories_savee={
    'a':'angry',
    'h':'happy',
    'n':'neutral',
    'sa':'sad',
    'su':'surprise',#和ps(PleasantSurprise)相关但是有一定区别
    'd':'disgust',
    'f':'fear',
}
# 配置当前(默认变量)
f_config_def = MCM
e_config_def = HNS

ava_algorithms = ['BEST_ML_MODEL', 'SVC', 'RandomForestClassifier', 'MLPClassifier', 'KNeighborsClassifier','BaggingClassifier','GradientBoostingClassifier','RNN']

#
def extend_emotion_names(emotion_first_letters):
    emotion_first_letters=emotion_first_letters.upper()
    res=[emotions_extend_dict.get(e) for e in emotion_first_letters]
    return res




if __name__=="__main__":
    # res=extend_names("HNS")
    # print(res)
    res=get_f_config_dict(f_config_def)
    print(res)


# layout=[
#     [sg.T(f"chose the db for {train}:")],[sg.LB(values=ava_dbs,size=(15,5))]
# ]

