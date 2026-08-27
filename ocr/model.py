import os
import numpy as np
from keras.layers import (Input, Conv2D, MaxPooling2D, ZeroPadding2D,
                         BatchNormalization, Permute, TimeDistributed, 
                         Flatten, Bidirectional, GRU, Dense, Lambda)
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
import keys_ocr

# Define CTC loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]  # Remove first two frames for CTC
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Model architecture
def get_model(height, nclass):
    rnnunit = 256
    input_tensor = Input(shape=(height, None, 1), name='the_input')

    # CNN layers
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = ZeroPadding2D(padding=(0, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = ZeroPadding2D(padding=(0, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid')(x)

    # Reshape for RNN
    x = Permute((2, 1, 3))(x)
    x = TimeDistributed(Flatten())(x)

    # RNN layers
    x = Bidirectional(GRU(rnnunit, return_sequences=True))(x)
    x = Dense(rnnunit, activation='linear')(x)
    x = Bidirectional(GRU(rnnunit, return_sequences=True))(x)
    y_pred = Dense(nclass, activation='softmax')(x)

    # Create model for training
    basemodel = Model(inputs=input_tensor, outputs=y_pred)

    # Define inputs for CTC loss
    labels = Input(name='the_labels', shape=[None, ], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    
    # Compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    return model, basemodel

# Load model
characters = keys_ocr.alphabet[:]
modelPath = os.path.join(os.getcwd(), "ocr/ocr0.2.h5")
height = 32
nclass = len(characters) + 1

if os.path.exists(modelPath):
    model, basemodel = get_model(height, nclass)
    basemodel.load_weights(modelPath)

def predict(im):
    """
    Input an image and return the recognized result from the keras model.
    """
    im = im.convert('L')  # Convert image to grayscale
    scale = im.size[1] / 32.0
    w = int(im.size[0] / scale)
    im = im.resize((w, 32))
    
    img = np.array(im).astype(np.float32) / 255.0
    X = img.reshape((32, w, 1))
    X = np.array([X])
    
    # Predict
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, 2:, :]  # Remove first two frames
    out = decode(y_pred)

    # Clean output
    out = clean_output(out)
    return out

def decode(pred):
    charactersS = characters + ' '  # Add space character
    t = pred.argmax(axis=2)[0]
    char_list = []
    n = len(characters)
    
    for i in range(len(t)):
        if t[i] != n and (i == 0 or t[i] != t[i - 1]):  # Avoid duplicates
            char_list.append(charactersS[t[i]])
    
    return ''.join(char_list)

def clean_output(out):
    while out and out[0] == '。':
        out = out[1:]  # Remove leading punctuation
    return out