##
# from ..emotion_recognition import EmotionRecognition
from emotion_recognition import EmotionRecognizer

from MetaPath import meta_pairs,emodb,ravdess
from sklearn.svm import SVC

from tkinter import *
from tkinter import ttk

import os

os.chdir("..")
cwd=os.getcwd()
print(f"{cwd=}")
#variables:
f_config=['mfcc','mel']
e_config=['angry','sad']
def initialze_er():
    # rec = EmotionRecognizer(model=my_model,e_config=AHNPS,f_config=f_config_def,test_dbs=[ravdess],train_dbs=[ravdess], verbose=1)
    meta_dict = {"train_dbs": emodb, "test_dbs": ravdess}
    # use SVC as a demo
    my_model = SVC(C=0.001, gamma=0.001, kernel="poly")
    my_model = None
    rec = EmotionRecognizer(
        model=my_model, e_config=e_config, f_config=f_config, **meta_dict, verbose=1
    )
    rec.train()
    score=rec.test_score()
    print(score,"@{score}")
    return rec

##
root=Tk()
frame=ttk.Frame(root)
frame.grid()

btn=ttk.Button(frame,text="initialize the ER object")
btn.grid()
btn.configure(command=initialze_er)

root.mainloop()

