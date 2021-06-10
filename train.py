#type:ignore
from model import model
from dataset import MySequence
import tensorflow as tf
import keras
import pickle

wavenet = model(
    out_channels=512, time_length=8000, residual_channels=128, residual_layers=20, 
    residual_stacks=2, skip_out_channels=128, kernel_size=3, dropout=0.05, 
    cin_channels=-1, cin_time_length=-1, gin_channels=-1, gate_channels=256,
    num_speakers=1, use_cin=False, use_gin=False).wavenet()
wavenet.compile("rmsprop", tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

# wavenet.summary()

seq = MySequence("dataset.pickle", 2, False)
#/content/drive/MyDrive/DL/Models
#"/content/drive/MyDrive/DL/history/history.dat"
modelchk = tf.keras.callbacks.ModelCheckpoint(filepath="/content/drive/MyDrive/DL/Models", monitor="val_accuracy", save_best_only=True, mode="max")
history = wavenet.fit(seq, epochs=25, validation_data=MySequence("dataset.pickle", 2, True), callbacks=[modelchk])
with open("/content/drive/MyDrive/DL/history/history.dat", "wb") as f:
    pickle.dump(history.history, f)