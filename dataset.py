#type:ignore
import librosa
import soundfile
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pickle
import keras
from tensorflow.python.ops.gen_batch_ops import batch

def mu_law(x, mu):
    y = np.sign(x) * np.log(1 + mu * np.abs(x)) / (np.log(1 + mu))
    y *= mu
    y = np.round(y)
    return y
    
def inv_mu_law(x,mu):
    y = x / mu
    return np.sign(y) * (1./mu) * np.ceil((1+mu)**np.abs(y) - 1)

def make_dataset():
    dir = r"dataset_folder_here"
    ds = []
    for ind, file in enumerate(glob(dir + r"\*")):
        print(file)
        wav,sr = librosa.load(file, sr=24000)
        mu = 255
        encoded = mu_law(wav, mu) + mu
        ds.append(encoded)
        # fp = np.memmap("dataset.dat", dtype="float32", mode="r+", shape=(100*20, 8000, 512))
        # fp[ind*20:(ind+1)*20]=np.array(tmp)
        # del fp

    with open("dataset.pickle", mode="wb") as f:
        pickle.dump(ds, f)

class MySequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, is_test):
        self.batch_size = batch_size
        self.num_samples = 8000
        self.mu = 255
        self.is_test = is_test
        with open("dataset.pickle", "rb") as f:
            self.data = pickle.load(f)
    def __len__(self):
        return 10*20 if self.is_test else 90*20
    def __getitem__(self, index):
        x = []
        y = []
        for i in range(self.batch_size):
            sentence = np.random.randint(90) if not self.is_test else np.random.randint(90,100)
            wav = self.data[sentence]
            head = np.random.randint(len(wav) - self.num_samples-1)
            wav = to_categorical(wav[head:head+self.num_samples+1], (self.mu+1)*2)
            x.append(wav[:-1])
            y.append(wav[1:])
        return np.array(x), np.array(y)