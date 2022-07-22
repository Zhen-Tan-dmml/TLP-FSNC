from utils import process
import numpy as np
from os.path import join
import pickle5 as pickle

def load_object(filename):
    with open(filename, 'rb') as fin:
        obj = pickle.load(fin)
    return obj
dataset = "citeseer"
adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
# rowsum = np.array(features.sum(1))

# features, _ = process.preprocess_features(features)
features = load_object(join("./fs_data/CoraFull", "features.pk"))