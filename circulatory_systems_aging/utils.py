import torch
import pickle

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def pickle_save(file, data):
    with open(file, "wb") as h:
        pickle.dump(data, h)


def pickle_load(file):
    with open(file, "rb") as h:
        data = pickle.load(h)
    return data
