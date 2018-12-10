# *-* coding: utf-8 *-*

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib


class CharNgramClassifier(BaseEstimator):
    def __init__(self, vect=TfidfVectorizer(analyzer="char", ngram_range=(3, 8)), model=GradientBoostingClassifier()):
        self.vect = vect
        self.model = model

    def fit(self, X, y):
        x_documents = [doc["text"] for doc in X]
        self.vect.fit(x_documents)
        x_encoded = self.vect.transform(x_documents)
        self.model.fit(x_encoded, y)
        return self

    def predict(self, X):
        x_documents = [doc["text"] for doc in X]
        x_encoded = self.vect.transform(x_documents)
        return self.model.predict(x_encoded)

    def predict_proba(self, X):
        x_documents = [doc["text"] for doc in X]
        x_encoded = self.vect.transform(x_documents)
        return self.model.predict_proba(x_encoded)

    def save_model(self, model_path):
        return joblib.dump(self, model_path)

    @staticmethod
    def load_model(model_path):
        return joblib.load(model_path)