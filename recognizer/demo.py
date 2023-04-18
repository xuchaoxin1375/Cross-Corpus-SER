from recognizer.basic import EmotionRecognizer
from sklearn.svm import SVC
# init a model, let's use SVC
my_model = SVC()
# pass my model to EmotionRecognizer instance
# and balance the dataset
rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)

##
rn=rec.get_meta_path()
print(rn)
##
# # train the model
# rec.train()


# # check the test accuracy for that model
# print("Test score:", rec.test_score())
# # check the train accuracy for that model
# print("Train score:", rec.train_score())