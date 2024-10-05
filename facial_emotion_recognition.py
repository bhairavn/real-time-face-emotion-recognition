import os
import cv2
import numpy as np
import pandas as pd
import math
import joblib
import seaborn as sns
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils
raw_data = pd.read_csv("/content/drive/MyDrive/fer2013.csv")
print("Shape of the dataset",raw_data.shape)
print("The first few rows of the table")
print(raw_data.head())
print("Unique emotion labels",raw_data.emotion.unique())

emotion_label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness',
                         4: 'sadness', 5: 'surprise', 6: 'neutral'}


                 raw_data.emotion.value_counts()
sns.countplot(raw_data.emotion)
pyplot.show()
print(math.sqrt(len(raw_data.pixels[0].split(' '))))
fig = pyplot.figure(1, (14, 14))# figure of height and width 14 inches

k = 0
for label in sorted(raw_data.emotion.unique()):
    for j in range(7):
        px = raw_data[raw_data.emotion==label].pixels.iloc[k]
        px = np.array(px.split(' ')).reshape(48, 48).astype('float32')

        k += 1
        ax = pyplot.subplot(7, 7, k)
        ax.imshow(px, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(emotion_label_to_text[label])
        pyplot.tight_layout()
INTERESTED_LABELS = [0,3, 4, 5]
raw_data = raw_data[raw_data.emotion.isin(INTERESTED_LABELS)]
raw_data.shape
img_array = raw_data.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48,
                                                      48, 1).astype('float32'))
# apply passes the function and applies to every single row
img_array = np.stack(img_array, axis=0)
#join a img_array of all the examples
img_array.shape
# paramenter for gabor filters
params1 = {'ksize':(18, 18), 'sigma':1.5, 'theta': 45,
           'lambd':5, 'gamma':1.5, 'psi':0}
params2 = {'ksize':(18, 18), 'sigma':1.5, 'theta': 135,
           'lambd':5, 'gamma':1.5, 'psi':0}
filter1 = cv2.getGaborKernel(**params1)
filter2 = cv2.getGaborKernel(**params2)
filter1 = tf.expand_dims(filter1, 2)
filter1 = tf.expand_dims(filter1, 3)
filter2 = tf.expand_dims(filter2, 2)
filter2 = tf.expand_dims(filter2, 3)
img_array = tf.nn.conv2d(img_array, filter1, strides=[1, 1, 1, 1], padding='SAME')
print("First filter applied")
img_array = tf.nn.conv2d(img_array, filter2, strides=[1, 1, 1, 1], padding='SAME')
print("Second filter applied")
#img_array.shape
'''img_array = tf.make_tensor_proto(img_array)
img_array = tf.make_ndarray(img_array)

print(type(img_array))
print(img_array.shape)'''
le = LabelEncoder()
img_labels = le.fit_transform(raw_data.emotion)
img_labels = np_utils.to_categorical(img_labels)
img_labels.shape
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels, shuffle=True, stratify=img_labels,test_size=0.1, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
del raw_data
del img_array
del img_labels
img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]
X_train = X_train / 255.
X_valid = X_valid / 255.
print(X_train.shape)
print(y_train)
def build_net(optim):
    """
    This is a Deep Convolutional Neural Network (DCNN). For generalization purpose I used dropouts in regular intervals.
    I used `ELU` as the activation because it avoids dying relu problem but also performed well as compared to LeakyRelu
    atleast in this case. `he_normal` kernel initializer is used as it suits ELU. BatchNormalization is also used for better
    results.
    """

    net = Sequential(name='DCNN')

    net.add(
        Conv2D(
            filters=64, # number of filters
            kernel_size=(5,5), # the filter size or the kernel size
            input_shape=(img_width, img_height, img_depth),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )
    )
    # applying batch normalization
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
    net.add(BatchNormalization(name='batchnorm_2'))

    # applying maxpool
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
    net.add(Dropout(0.6, name='dropout_1'))

    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
    net.add(BatchNormalization(name='batchnorm_4'))

    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
    # applying dropout regularization to avoid overfitting
    net.add(Dropout(0.5, name='dropout_2'))

    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
    net.add(BatchNormalization(name='batchnorm_5'))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )
    )
    net.add(BatchNormalization(name='batchnorm_6'))

    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
    net.add(Dropout(0.4, name='dropout_3'))

    net.add(Flatten(name='flatten'))

    net.add(
        Dense(
            128,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_7'))

    net.add(Dropout(0.4, name='dropout_4'))

    net.add(
        Dense(
            num_classes,
            activation='softmax',
            name='out_layer'
        )
    )
    # the evaluation criteria for loss is cross-entropy

    net.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy']
    )

    net.summary()

    return net
'''def build_net(optim):
  net = Sequential(name='DCNN')
  #This is the CNN from the research paper
  net.add(
        Conv2D(
            filters=6,
            kernel_size=(3,3),
            input_shape=(img_width, img_height, img_depth),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )
    )
  #net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
  net.add(
        Conv2D(
            filters=16,
            kernel_size=(3,3),
            input_shape=(img_width, img_height, img_depth),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
  #net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
  net.add(
        Conv2D(
            filters=120,
            kernel_size=(3,3),
            input_shape=(img_width, img_height, img_depth),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
  net.add(Dropout(0.5, name='dropout_1'))
  net.add(Flatten(name='flatten'))
  net.add(
        Dense(
            84,
            activation='relu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
  net.add(Dropout(0.5, name='dropout_2'))
  net.add(
        Dense(
            num_classes,
            activation='softmax',
            name='out_layer'
        )
    )

  net.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy']
    )

  net.summary()

  return net'''

"""
I used two callbacks one is `early stopping` for avoiding overfitting training data
and other `ReduceLROnPlateau` for learning rate.
"""

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=10,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,# set to 0.1 or 0.2
    patience=7, # set to 5-10
    min_lr=1e-6, # might want to increase this, originally was 1e-7
    verbose=1, # keep this as 1
)

callbacks = [
    early_stopping,
    lr_scheduler,
]
# Generating new data
train_datagen = ImageDataGenerator(
    rotation_range=15,#15,
    width_shift_range=0.15,#0.15,
    height_shift_range=0.15,#0.15,
    shear_range=0.15,#0.15,
    zoom_range=0.15,#0.15,
    horizontal_flip=True,#True,
)
train_datagen.fit(X_train)
print(X_train.shape)
batch_size = 32 #batch size of 32 performs the best.
epochs = 100
optims = [
    optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    optimizers.Adam(0.001),
]


model = build_net(optims[1])
#print(X_train[0])
history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    use_multiprocessing=True
)
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

model.save("model.h5")
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
#model.save('model.h5')
model_file = drive.CreateFile({'title' : 'model.h5'})
model_file.SetContentFile('model.h5')
model_file.Upload()
# download to google drive
drive.CreateFile({'id': model_file.get('id')})
yhat_valid = model.predict_classes(X_valid)


print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid, axis=1) != yhat_valid)}\n\n')
print(classification_report(np.argmax(y_valid, axis=1), yhat_valid))
