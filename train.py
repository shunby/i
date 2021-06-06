from model import model
from dataset import MySequence
import tensorflow as tf
import keras
import pickle

wavenet = model(
    out_channels=512, time_length=8000, residual_channels=128, residual_layers=24, 
    residual_stacks=4, skip_out_channels=128, kernel_size=3, dropout=0.05, 
    cin_channels=-1, cin_time_length=-1, gin_channels=-1, gate_channels=256,
    num_speakers=1, use_cin=False, use_gin=False).wavenet()
wavenet.compile("rmsprop", tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

seq = MySequence(10, False)
modelchk = tf.keras.callbacks.ModelCheckpoint(filepath="/content/drive/MyDrive/DL/Models", monitor="val_accuracy", save_best_only=True, mode="max")
history = wavenet.fit(seq, epochs=50, validation_data=MySequence(10, True), callbacks=[modelchk])
with open("/content/drive/MyDrive/DL/history/history.dat", "wb") as f:
    pickle.dump(history, f)