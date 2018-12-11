import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import MultiLabelBinarizer
import keras
from keras.layers import Conv1D, MaxPool1D, LSTM, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from uuid import uuid4


class MorphVectorizer(BaseEstimator):
    def __init__(self, sep="|"):
        self.morph_enc = MultiLabelBinarizer()
        self.sep = sep

    def fit(self, X, y=None):
        l_x_tags = []
        for doc in X:
            for sent in doc["sentences"]:
                for word in sent:
                    word_morph = [word["pos"]] + word["grm"].split(self.sep)
                    l_x_tags.append(word_morph)
        self.morph_enc.fit(l_x_tags)

    def transform(self, X, y=None):
        l_docs = []
        for doc in X:
            l_doc = []
            for sent in doc["sentences"]:
                for word in sent:
                    word_morph = [word["pos"]] + word["grm"].split(self.sep)
                    l_doc.append(word_morph)
            arr_doc = self.morph_enc.transform(l_doc)
            l_docs.append(arr_doc)

        max_doc_len = max([doc.shape[0] for doc in l_docs])
        word_dim = len(self.morph_enc.classes_)
        for doc_ind, doc in enumerate(l_docs):
            doc_padded = np.zeros((max_doc_len, word_dim))
            doc_padded[:len(doc)] = doc
            l_docs[doc_ind] = np.array(doc_padded, copy=True)

        l_docs = np.array(l_docs)
        return l_docs

    def fit_transofrm(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class NNModelSimple(BaseEstimator):
    def __init__(self, model_path, batch_size=32):
        self.y_label_enc = MultiLabelBinarizer()
        self.morph_enc = MorphVectorizer()
        self.model = Sequential()
        self.model_path = model_path
        self.batch_size = batch_size
        self.h = keras.callbacks.History

    def fit(self, X, y):
        # Для разделения на тренировочное и валидационное множества по ID пользователя
        user_ids = []
        for doc in X:
            doc_user_id = doc["meta"].get("user_id", str(uuid4()))
            if "user_id" not in doc["meta"]:
                doc["meta"]["user_id"] = doc_user_id
            user_ids.append(doc_user_id)

        tr_user_ids, vl_user_ids = train_test_split(np.unique(user_ids), train_size=0.9, test_size=0.1)

        y = np.array([str(val) for val in y])
        self.y_label_enc.fit(y)
        self.morph_enc.fit(X)
        X = np.array(X)

        X_tr = np.array([doc for doc, user_id in zip(X, user_ids) if user_id in tr_user_ids])
        X_vl = np.array([doc for doc, user_id in zip(X, user_ids) if user_id in vl_user_ids])
        y_tr = np.array([doc for doc, user_id in zip(y, user_ids) if user_id in tr_user_ids])
        y_vl = np.array([doc for doc, user_id in zip(y, user_ids) if user_id in vl_user_ids])
        y_tr_enc = self.y_label_enc.transform(y_tr)
        y_vl_enc = self.y_label_enc.transform(y_vl)

        self.model = self.keras_model(len(self.y_label_enc.classes_), len(self.morph_enc.morph_enc.classes_))
        save_model = ModelCheckpoint(os.path.join(self.model_path, "model.hdf5"), save_best_only=True)
        early_stop = EarlyStopping(patience=30)
        self.h = self.model.fit_generator(self.batch_gen(X_tr, y_tr_enc), self.get_n_steps(len(X_tr), self.batch_size),
                                          validation_data=self.batch_gen(X_vl, y_vl_enc),
                                          validation_steps=self.get_n_steps(len(X_vl), self.batch_size),
                                          verbose=1, epochs=300, callbacks=[save_model, early_stop]).history
        return self

    def save_model(self, model_path):
        joblib.dump(self.y_label_enc, os.path.join(model_path, "y_label_enc.pkl"))
        joblib.dump(self.morph_enc, os.path.join(model_path, "morph_enc.pkl"))
        d_model_params = {}
        for k, v in self.__dict__.items():
            if k not in ["y_label_enc", "morph_enc", "model"]:
                d_model_params[k] = v
        with open(os.path.join(model_path, "model_params.json"), "w") as f:
            return json.dump(d_model_params, f)

    def load_model(self, model_path):
        with open(os.path.join(model_path, "model_params.json"), "r") as f:
            d_model_params = json.load(f)
        for k, v in d_model_params.items():
            self.k = v
        self.y_label_enc = joblib.load(os.path.join(model_path, "y_label_enc.pkl"))
        self.morph_enc = joblib.load(os.path.join(model_path, "morph_enc.pkl"))
        self.model = keras.models.load_model(os.path.join(model_path, "model.hdf5"))
        return self

    def predict(self, X, y=None):
        X = np.array(X)
        pred = self.model.predict_generator(self.batch_gen(X, None),
                                            steps=self.get_n_steps(len(X), self.batch_size),
                                            verbose=1)
        res = np.zeros(pred.shape)
        val_ind = 0
        for val in pred:
            res[val_ind, np.argmax(val)] = 1.
            val_ind += 1
        res = self.y_label_enc.inverse_transform(res)
        res = np.array([val[0] for val in res])
        return res

    def batch_gen(self, X, y=None, batch_size=32, shuffle=False):
        inds = np.arange(0, len(X))
        while True:
            if shuffle:
                np.random.shuffle(inds)

            for start_ind in np.arange(0, len(inds), batch_size):
                batch_inds = inds[start_ind:start_ind + batch_size]
                x_batch =  X[batch_inds]
                x_batch_enc = self.morph_enc.transform(x_batch)
                if y is not None:
                    y_batch = y[batch_inds]
                    yield x_batch_enc, y_batch
                else:
                    yield x_batch_enc

    @staticmethod
    def get_n_steps(seq_len, batch_size=32):
        res = int(seq_len/batch_size)
        if seq_len % batch_size != 0:
            res += 1
        return res

    @staticmethod
    def keras_model(out_dim, tags_dim):
        model_hidden = Sequential()
        model_hidden.add(Conv1D(128, 2, activation="relu", padding="same", input_shape=[None, tags_dim]))
        model_hidden.add(MaxPool1D(2, padding="same"))
        model_hidden.add(Conv1D(128, 2, activation="relu", padding="same"))
        model_hidden.add(MaxPool1D(2, padding="same"))
        model_hidden.add(Conv1D(128, 2, activation="relu", padding="same"))
        model_hidden.add(MaxPool1D(2, padding="same"))
        model_hidden.add(LSTM(128))

        model = Sequential()
        model.add(model_hidden)
        model.add(Dropout(0.5))
        model.add(Dense(out_dim, activation="softmax"))

        model.compile("adam", "mse", metrics=["accuracy"])

        return model