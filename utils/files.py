import pickle


def load_data(filepath):
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    return data


def save_data(data, filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)
