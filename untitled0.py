
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:19:51 2020

@author: user
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from sklearn.model_selection import train_test_split

labels_df = pd.read_csv('labels.csv')
labels_df.head()

labels = np.array(labels_df[' hemorrhage'].tolist())
labels

files = sorted(glob.glob('head_ct/*.png'))
images = np.array([cv2.imread(path) for path in files])

print(images.shape)     #200, 
images[0].shape         # 957, 821, 3
images[10].shape        # 285, 247, 3

plt.figure(figsize=(10, 10))
for i in range(1, 11):
    plt.subplot(5, 5, i)
    plt.imshow(images[i])
    
images = np.array([cv2.resize(image, (256, 256)) for image in images])
images.shape   #200, 256, 256, 3 


plt.figure(figsize=(10, 10))
for i in range(1, 11):
    plt.subplot(5, 5, i)
    plt.imshow(images[i])

print(labels)  

train_ratio = 0.75           #verisetinin %75i train için
validation_ratio = 0.15      #verisetinin %15i validasyon için
test_ratio = 0.10            #verisetinin %10i test için
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=1 - train_ratio)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
print(X_train.shape, Y_train.shape)   
print(X_val.shape, Y_val.shape)
print(X_test.shape, Y_test.shape)
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

train_image_data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.,
    zoom_range=0.05,
    rotation_range=0,
    width_shift_range=0.05,
    height_shift_range=0.05
)
validation_image_data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.,
    zoom_range=0.05,
    rotation_range=0,
    width_shift_range=0.05,
    height_shift_range=0.05)

def create_model(input_shape):    #model kurma
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(GlobalAveragePooling2D()) #GAP, her özellik haritasındaki ortalama etkinleştirme değerini alır ve tek boyutlu bir tensör döndürür.
    #GAP katmanı, modelin öznitelik çıkarma kısmını tamamlar.
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1, activation='sigmoid'))  #sigmoid fonk. 2li sonuç olduğu için
    return model

model = create_model((256, 256, 3))
model.summary() #mimariyi göster

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) #en iyi modelin eğitim sürecinde otomatik olarak bir dizine kaydedilmesini sağlayan 
mc = ModelCheckpoint('weights.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)   # en iyi modeli otomatik olarak kaydet
callbacks = [es, mc]
history = model.fit_generator(train_image_data.flow(X_train, Y_train, batch_size=4),
                              steps_per_epoch=X_train.shape[0] // 4,
                              validation_data=validation_image_data.flow(X_val, Y_val, batch_size=8),
                              validation_steps=X_val.shape[0] // 8,
                              callbacks=callbacks,
                              epochs=20)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim([.5, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim([0, 1])
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()

test_image_data = ImageDataGenerator(rescale=1./255)

test_generator = test_image_data.flow(X_test, batch_size=1, shuffle=False)
test_pred = model.predict_generator(test_generator, X_test.shape[0], verbose = 1)   #variable explorer

test_pred2 = (test_pred > 0.5)   
Y_test2=(Y_test > 0.5)
acc = accuracy_score(Y_test2, test_pred2)
print('Acc: %', acc*100)
























