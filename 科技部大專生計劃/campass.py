import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import json, jieba, requests
from udicOpenData.dictionary import *
from udicOpenData.stopwords import rmsw
from sklearn.model_selection import train_test_split


# load dataset
# split into input (X) and output (Y) variables
dataset = json.load(open('newcampass.json', 'r'))
X = np.array([sum([np.array(requests.get('http://udiclab.cs.nchu.edu.tw/kem/vector?keyword={}'.format(j)).json()["value"]) for j in jieba.cut(i['raw'])]) for i in dataset])

Y = np.array([np.array((i['feedback_knowledgeable'], i['feedback_GPA'], i['feedback_freedom'], i['feedback_easy'], i['feedback_FU'])) for i in dataset])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(units=200, activation='softmax', input_shape=(400, )))
model.add(Dense(units=50, activation='softmax'))
model.add(Dense(units=50, activation='softmax'))
model.add(Dense(units=50, activation='softmax'))
model.add(Dense(units=30, activation='softmax'))
model.add(Dense(units=20, activation='softmax'))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
train_history = model.fit(X_train, y_train, epochs=5000, batch_size=32, validation_split=0.001, verbose=2)
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
json.dump(train_history.history, open('train_history.json', 'w'))

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    axes = plt.gca()
    axes.set_ylim([0.419,0.44])
    plt.plot(train_history[train])
    plt.plot(train_history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


import json
show_train_history(json.load(open('train_history.json', 'r')), 'mean_absolute_error','val_mean_absolute_error')