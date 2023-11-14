import pickle

import wget

from .autoencoder import CustomAE as AutoEncoder
from .load_pretrained_model import LoadPretrainedModel
from .models_genesis import UNet3D as ModelsGenesisUNet3D
from .resnet50 import resnet50


def get_linear_classifier(weights_path=None, download_url="https://www.dropbox.com/s/77zg2av5c6edjfu/task3.pkl?dl=1"):
    if weights_path is None:
        weights_path = "/tmp/linear_model.pkl"
        wget.download(download_url, out=weights_path)

    return pickle.load(open(weights_path, "rb"))
