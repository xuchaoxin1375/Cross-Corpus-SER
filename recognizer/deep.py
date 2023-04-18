import os
# disable keras loggings
import sys

from config.EF import e_config_def

stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_absolute_error)
# import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (LSTM, Dense, Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from config.EF import validate_emotions
from recognizer.basic import EmotionRecognizer
# from ER import EmotionRecognizer
from config.MetaPath import get_first_letters
from audio.core import extract_feature, get_dropout_str


class DeepEmotionRecognizer(EmotionRecognizer):
    """
    本类继承自基础的EmotionRecognizer
    The Deep Learning version of the Emotion Recognizer.
    This class uses RNN (LSTM, GRU, etc.) and Dense layers.

    """

    def __init__(
        self,
        train_dbs=None,
        test_dbs=None,
        dropout=None,
        e_config=None,
        f_config=None,
        optimizer="adam",
        loss="categorical_crossentropy",
        batch_size=64,
        classification_task=True,
        epochs=200,
        **kwargs,
    ):
        """
        数据集和特征相关参数初始化
        默认识别3种情感HNS

        params:
        -
            emotions (list): list of emotions to be used. Note that these emotions must be available in
                RAVDESS & EMODB Datasets, available nine emotions are the following:
                    'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps' ( pleasant surprised ), 'boredom'.
                Default is ["sad", "neutral", "happy"].
            ravdess (bool): whether to use RAVDESS Speech datasets, default is True.
            emodb (bool): whether to use EMO-DB Speech dataset, default is True.
            custom_db (bool): whether to use custom Speech dataset that is located in `data/train-custom`
                and `data/test-custom`, default is True.
            ravdess_name (str): the name of the output CSV file for RAVDESS dataset, default is "ravdess.csv".
            emodb_name (str): the name of the output CSV file for EMO-DB dataset, default is "emodb.csv".
            custom_db_name (str): the name of the output CSV file for the custom dataset, default is "custom.csv".

            features (list): list of speech features to use, default is ["mfcc", "chroma", "mel"]
                (i.e MFCC, Chroma and MEL spectrogram ).
            classification (bool): whether to use classification or regression, default is True.
            balance (bool): whether to balance the dataset ( both training and testing ), default is True.
            verbose (bool/int): whether to print messages on certain tasks.
            ==========================================================
            RNN超参数
            Model params
            n_rnn_layers (int): number of RNN layers, default is 2.
            cell (keras.layers.RNN instance): RNN cell used to train the model, default is LSTM.
            rnn_units (int): number of units of `cell`, default is 128.
            n_dense_layers (int): number of Dense layers, default is 2.
            dense_units (int): number of units of the Dense layers, default is 128.
            dropout (list/float): dropout rate,
                - if list, it indicates the dropout rate of each layer.
                - if float, it indicates the dropout rate for all layers.
                Default is 0.3.
            ==========================================================
            训练方式参数
            Training params
            batch_size (int): number of samples per gradient update, default is 64.
            epochs (int): number of epochs, default is 200.
            optimizer (str/keras.optimizers.Optimizer instance): optimizer used to train, default is "adam"(自适应矩估计).
            loss (str/callback from keras.losses): loss function that is used to minimize during training,
                default is "categorical_crossentropy" for classification and "mean_squared_error" for
                regression.
        """
        # init EmotionRecognizer
        super().__init__(
            e_config=e_config,
            f_config=f_config,
            train_dbs=train_dbs,
            test_dbs=test_dbs,
            classification_task=classification_task,
            **kwargs,
        )

        # 放在靠前的位置
        self.int2emotions = {i: e for i, e in enumerate(self.e_config)}
        self.emotions2int = {v: k for k, v in self.int2emotions.items()}

        self.n_rnn_layers = kwargs.get("n_rnn_layers", 2)
        self.n_dense_layers = kwargs.get("n_dense_layers", 2)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.dense_units = kwargs.get("dense_units", 128)
        self.cell = kwargs.get("cell", LSTM)

        # list of dropouts of each layer
        # must be len(dropouts) = n_rnn_layers + n_dense_layers
        self.dropout = dropout if dropout else 0.3
        self.dropout = (
            self.dropout
            if isinstance(self.dropout, list)
            else [self.dropout] * (self.n_rnn_layers + self.n_dense_layers)
        )
        # number of classes ( emotions )
        self.output_dim = len(self.e_config)
  

        # optimization attributes
        self.optimizer = optimizer if optimizer else "adam"
        self.loss = loss if loss else "categorical_crossentropy"

        # training attributes
        self.batch_size = batch_size if batch_size else 64
        self.epochs = epochs if epochs else 200

        # the name of the model
        self.model_name = ""
        self._update_model_name()

        # init the model
        self.model = None

        # compute the input length
        self._compute_input_length()

        # boolean attributes
        self.model_created = False

        self.model_path = f"results/{self.model_name}"

    def _update_model_name(self):
        """
        Generates a unique model name based on parameters passed and put it on `self.model_name`.
        This is used when saving the model.
        """

        emotions_str = get_first_letters(self.e_config)
        # 'c' for classification & 'r' for regression
        problem_type = "c" if self.classification_task else "r"
        dropout_str = get_dropout_str(
            self.dropout, n_layers=self.n_dense_layers + self.n_rnn_layers
        )
        self.model_name = f"{emotions_str}-{problem_type}-{self.cell.__name__}-layers-{self.n_rnn_layers}-{self.n_dense_layers}-units-{self.rnn_units}-{self.dense_units}-dropout-{dropout_str}.h5"

    def _model_exists(self):
        """
        Checks if model already exists in disk, returns the filename,
        and returns `None` otherwise.
        """
        model_path = self.model_path
        return model_path if os.path.isfile(model_path) else None

    def _compute_input_length(self):
        """
        Calculates the input shape to be able to construct the model.
        """
        if not self.data_loaded:
            self.load_data()
        self.input_length = self.X_train[0].shape[1]

    def _verify_emotions(self):
        validate_emotions(self.e_config)
        self.int2emotions = {i: e for i, e in enumerate(self.e_config)}
        self.emotions2int = {v: k for k, v in self.int2emotions.items()}
        print("self.int2emotions:", self.int2emotions)
        print("self.emotions2int:", self.emotions2int)

    def create_model(self):
        """
        Constructs the neural network based on parameters passed.
        """
        if self.model_created:
            # model already created
            return

        if not self.data_loaded:
            # if data isn't loaded yet, load it
            self.load_data()

        model = Sequential()

        # rnn layers
        for i in range(self.n_rnn_layers):
            if i == 0:
                # first layer
                model.add(
                    self.cell(
                        self.rnn_units,
                        return_sequences=True,
                        input_shape=(None, self.input_length),
                    )
                )
                model.add(Dropout(self.dropout[i]))
            else:
                # middle layers
                model.add(self.cell(self.rnn_units, return_sequences=True))
                model.add(Dropout(self.dropout[i]))

        if self.n_rnn_layers == 0:
            i = 0

        # dense layers
        for j in range(self.n_dense_layers):
            # if n_rnn_layers = 0, only dense
            if self.n_rnn_layers == 0 and j == 0:
                model.add(
                    Dense(self.dense_units, input_shape=(None, self.input_length))
                )
                model.add(Dropout(self.dropout[i + j]))
            else:
                model.add(Dense(self.dense_units))
                model.add(Dropout(self.dropout[i + j]))

        if self.classification_task:
            model.add(Dense(self.output_dim, activation="softmax"))
            model.compile(
                loss=self.loss, metrics=["accuracy"], optimizer=self.optimizer
            )
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(
                loss="mean_squared_error",
                metrics=["mean_absolute_error"],
                optimizer=self.optimizer,
            )

        self.model = model
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")

    def load_data(self):
        """
        Loads and extracts features from the audio files for the db's specified.
        And then reshapes the data.
        """
        super().load_data()
        # reshape X's to 3 dims
        X_train_shape = self.X_train.shape
        X_test_shape = self.X_test.shape
        self.X_train = self.X_train.reshape((1, X_train_shape[0], X_train_shape[1]))
        self.X_test = self.X_test.reshape((1, X_test_shape[0], X_test_shape[1]))

        if self.classification_task:
            # one-hot encode when its classification
            self.y_train = to_categorical(
                [self.emotions2int[str(e)] for e in self.y_train]
            )
            self.y_test = to_categorical(
                [self.emotions2int[str(e)] for e in self.y_test]
            )

        # reshape labels
        if(self.y_train is  None or self.y_test is  None): 
            raise ValueError("y_train and y_test must be array_like ")
        y_train_shape = self.y_train.shape
        y_test_shape = self.y_test.shape
        if self.classification_task:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], y_train_shape[1]))
            self.y_test = self.y_test.reshape((1, y_test_shape[0], y_test_shape[1]))
        else:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], 1))
            self.y_test = self.y_test.reshape((1, y_test_shape[0], 1))

    def train(self, override=False):
        """
        Trains the neural network.
        Params:
            override (bool): whether to override the previous identical model, can be used
                when you changed the dataset, default is False
        """
        # if model isn't created yet, create it
        if not self.model_created:
            self.create_model()

        # if the model already exists and trained, just load the weights and return
        # but if override is True, then just skip loading weights
        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return

        if not os.path.isdir("../results"):
            os.mkdir("../results")

        if not os.path.isdir("../logs"):
            os.mkdir("../logs")

        model_filename = self.model_path

        self.checkpointer = ModelCheckpoint(
            model_filename, save_best_only=True, verbose=1
        )
        self.tensorboard = TensorBoard(log_dir=os.path.join("../logs", self.model_name))

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_test, self.y_test),
            callbacks=[self.checkpointer, self.tensorboard],
            verbose=self.verbose,
        )

        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    def predict(self, audio_path):
        feature = extract_feature(audio_path, **self._f_config_dict).reshape(
            (1, 1, self.input_length)
        )
        if self.classification_task:
            prediction = self.model.predict(feature)
            prediction = np.argmax(np.squeeze(prediction))
            return self.int2emotions[prediction]
        else:
            return np.squeeze(self.model.predict(feature))

    def predict_proba(self, audio_path):
        if self.classification_task:
            feature = extract_feature(audio_path, **self._f_config_dict).reshape(
                (1, 1, self.input_length)
            )
            proba = self.model.predict(feature)[0][0]
            result = {}
            for prob, emotion in zip(proba, self.e_config):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError(
                "Probability prediction doesn't make sense for regression"
            )

    def test_score(self):
        # 测试集标签
        y_test = self.y_test[0]
        if self.classification_task:
            y_pred = self.model.predict(self.X_test)[0]
            y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
            y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
            return accuracy_score(y_true=y_test, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_test)[0]
            return mean_absolute_error(y_true=y_test, y_pred=y_pred)

    def train_score(self):
        y_train = self.y_train[0]
        if self.classification_task:
            y_pred = self.model.predict(self.X_train)[0]
            y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
            y_train = [np.argmax(y, out=None, axis=None) for y in y_train]
            return accuracy_score(y_true=y_train, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_train)[0]
            return mean_absolute_error(y_true=y_train, y_pred=y_pred)

    def confusion_matrix(self, percentage=True, labeled=True):
        """Compute confusion matrix to evaluate the test accuracy of the classification"""
        if not self.classification_task:
            raise NotImplementedError(
                "Confusion matrix works only when it is a classification problem"
            )
        y_pred = self.model.predict(self.X_test)[0]
        y_pred = np.array([np.argmax(y, axis=None, out=None) for y in y_pred])
        # invert from keras.utils.to_categorical
        y_test = np.array([np.argmax(y, axis=None, out=None) for y in self.y_test[0]])
        matrix = confusion_matrix(
            y_test, y_pred, labels=[self.emotions2int[e] for e in self.e_config]
        ).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(
                matrix,
                index=[f"true_{e}" for e in self.e_config],
                columns=[f"predicted_{e}" for e in self.e_config],
            )
        return matrix

    def count_samples_in_partition(self, emotion, partition):
        """Returns number data samples of the `emotion` class in a particular `partition`
        ('test' or 'train')
        """
        if partition == "test":
            if self.classification_task:
                # np.squeeze去除数组中所有维度大小为 1 的维度，从而将数组的维度降低。如果数组没有大小为 1 的维度，则不会有任何变化。
                #这里的y_test可能采用oneHotEncoder,因此可以如下计算
                y_test = np.array(
                    [
                        np.argmax(y, axis=None, out=None) + 1
                        for y in np.squeeze(self.y_test)
                    ]
                )
            else:
                y_test = np.squeeze(self.y_test)
            return len([y for y in y_test if y == emotion])
        elif partition == "train":
            if self.classification_task:
                y_train = np.array(
                    [
                        np.argmax(y, axis=None, out=None) + 1
                        for y in np.squeeze(self.y_train)
                    ]
                )
            else:
                y_train = np.squeeze(self.y_train)
            return len([y for y in y_train if y == emotion])

    def count_samples_by_class(self):
        """
        Returns a dataframe that contains the number of training
        and testing samples for all emotions
        """
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.e_config:
            n_train = self.count_samples_in_partition(self.emotions2int[emotion] + 1, "train")
            n_test = self.count_samples_in_partition(self.emotions2int[emotion] + 1, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)

        # get total
        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(
            data={"train": train_samples, "test": test_samples, "total": total},
            index=self.e_config + ["total"],
        )
    
    def get_training_and_testing_samples_per_emotion(self):
        """
        Returns a dataframe with the number of training and testing samples
        per emotion, as well as the total number of samples.
        """
        train_samples_per_emotion = []
        test_samples_per_emotion = []
        total_samples_per_emotion = []
        for emotion in self.e_config:
            n_train = self.count_samples_in_partition(self.emotions2int[emotion] + 1, "train")
            n_test = self.count_samples_in_partition(self.emotions2int[emotion] + 1, "test")
            train_samples_per_emotion.append(n_train)
            test_samples_per_emotion.append(n_test)
            total_samples_per_emotion.append(n_train + n_test)

        total_train_samples = sum(train_samples_per_emotion)
        total_test_samples = sum(test_samples_per_emotion)
        total_samples = total_train_samples + total_test_samples

        return pd.DataFrame(
            data={
                "Train Samples": train_samples_per_emotion,
                "Test Samples": test_samples_per_emotion,
                "Total Samples": total_samples_per_emotion,
            },
            index=self.e_config + ["Total"],
        )


    def get_random_emotion_index(self, emotion, partition="train"):
        """
        Returns random `emotion` data sample index on `partition`
        """
        if partition == "train":
            y_train = self.y_train[0]
            index = random.choice(list(range(len(y_train))))
            element = self.int2emotions[np.argmax(y_train[index])]
            while element != emotion:
                index = random.choice(list(range(len(y_train))))
                element = self.int2emotions[np.argmax(y_train[index])]
        elif partition == "test":
            y_test = self.y_test[0]
            index = random.choice(list(range(len(y_test))))
            element = self.int2emotions[np.argmax(y_test[index])]
            while element != emotion:
                index = random.choice(list(range(len(y_test))))
                element = self.int2emotions[np.argmax(y_test[index])]
        else:
            raise TypeError("Unknown partition, only 'train' or 'test' is accepted")

        return index



##
if __name__ == "__main__":
    from config.MetaPath import ravdess
    meta_dict = {
        "train_dbs":ravdess,
        "test_dbs":ravdess
    }
    print(meta_dict)
    
    der = DeepEmotionRecognizer(**meta_dict, emotions=e_config_def, verbose=0)
    # #train
    der.train()
    print("train_score",der.train_score())
    print("test_score",der.test_score())
    # print(der.train_meta_files,der.test_meta_files)
    # der.train(override=False)
    # print("Test accuracy score:", der.test_score() * 100, "%")
