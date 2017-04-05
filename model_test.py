# coding: utf-8
"""
    Test on 2013 CASIA competition data

"""

import sys
import os
import codecs
import random
import copy
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from keras.utils.np_utils import to_categorical
from keras import backend
from the_model import model_8

random.seed(888)
np.random.seed(888)
tf.set_random_seed(888)

IMG_SIZE = 96

Pred_Details = False
if len(sys.argv) > 1:
    Pred_Details = True

TEST_PATH = os.path.join("data", "test")
WEIGHTS_PATH = os.path.join("data", "weights08.h5")
LABELS_PATH = os.path.join("data", "labels.txt")

label_file = codecs.open(LABELS_PATH, "r", "UTF-8")
klasses = [a.strip() for a in label_file.readlines()]
label_file.close()


def top_predictions(n, pred):
    tops = []
    pred_copy = copy.copy(pred)
    for j in range(n):
        i = np.argmax(pred_copy)
        tops.append((klasses[i], "%.3f" % (pred_copy[i] * 100)))
        pred_copy[i] = 0

    return tops


label_pngs = []
for k, v in enumerate(klasses):
    for png in os.listdir(os.path.join(TEST_PATH, v)):
        label_pngs.append((k, v, png))

print("Total number of test samples:", len(label_pngs))
test_data = np.ndarray([len(label_pngs), IMG_SIZE, IMG_SIZE], dtype=np.uint8)
test_label = np.ndarray([len(label_pngs)], dtype=np.uint32)

i = 0
for label_png in label_pngs:
    fimg = open(os.path.join(TEST_PATH, label_png[1], label_png[2]), 'rb')
    test_data[i] = imread(fimg)
    test_label[i] = label_png[0]
    fimg.close()
    i += 1

x_test = test_data.reshape(test_data.shape[0],
                           test_data.shape[1],
                           test_data.shape[2],
                           1)
x_test = x_test.astype(np.float32)
x_test /= 255.0
y_test = to_categorical(test_label)

model = model_8(IMG_SIZE, len(klasses))
model.load_weights(WEIGHTS_PATH)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if Pred_Details:
    preds = model.predict(x_test)
    for k, v in enumerate(preds):
        p = np.argmax(v)
        if p != test_label[k]:
            print("Wrong prediction: top3:", top_predictions(3, v),
                  "label/file:", label_pngs[k][1], label_pngs[k][2])
else:
    loss, acc = model.evaluate(x_test, y_test, batch_size=64)
    print("Loss:", loss)
    print("Accuracy:", acc)

backend.clear_session()
