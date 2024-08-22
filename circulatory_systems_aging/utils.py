import pickle


def pickle_save(file, data):
    with open(file, "wb") as h:
        pickle.dump(data, h)
