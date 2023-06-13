import pickle

import wget

from .resnet50 import resnet50


def get_linear_classifier():
    wget.download("https://www.dropbox.com/s/77zg2av5c6edjfu/task3.pkl?dl=1", out="/tmp/linear_model.pkl")
    return pickle.load(open("/tmp/linear_model.pkl", "rb"))
