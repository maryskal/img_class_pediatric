from tensorflow.keras import backend as K
import os
import tensorflow as tf


def list_files(path):
    return [f for f in os.listdir(path) 
                if os.path.isfile(os.path.join(path, f))]

def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def custom_binary_loss(y_true, y_pred): 
    print(y_true)
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  # Cancels out when target is 1 
    term_1 = y_true * K.log(y_pred + K.epsilon()) # Cancels out when target is 0
    suma = term_0 + term_1
    return -K.mean(suma, axis=1)+K.std(suma, axis = 1)


def custom_binary_loss_2(y_true, y_pred):
    print(y_true)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    san = K.sum(1-y_true, axis = 0)
    enf = K.sum(y_true, axis = 0)
    dif_sanos = (1 - y_true) * K.abs(y_true-y_pred)  # Cancels out when target is 1 
    dif_sanos = K.sum(dif_sanos, axis = 0)/san
    dif_enf = y_true * K.abs(y_true-y_pred) # Cancels out when target is 0
    dif_enf = K.sum(dif_enf, axis = 0)/enf
    dif_enf = tf.where(tf.math.is_nan(dif_enf), tf.zeros_like(dif_enf), dif_enf)
    dif_sanos = tf.where(tf.math.is_nan(dif_sanos), tf.zeros_like(dif_sanos), dif_sanos)
    suma = dif_enf + dif_sanos + K.abs(dif_enf-dif_sanos)
    suma = tf.where(tf.math.is_nan(suma), tf.zeros_like(suma), suma)
    print('\n')
    print(suma)
    return K.mean(suma)+ K.std(suma)
