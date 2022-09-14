from tensorflow.keras import backend as K
from tensorflow import keras
import os

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


mask_model = keras.models.load_model('./modelos/mask_1.h5', 
                                    custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
sub_mask = keras.Model(inputs=mask_model.input, outputs=mask_model.layers[18].output)
sub_mask.trainable = False


def loss_mask(y_true, y_pred):
    y_pred = sub_mask(y_pred)
    y_true = sub_mask(y_true)
    return 0.6*abs(y_true - y_pred)


def MyLoss(y_true, y_pred):
    # Loss 1
    loss1 = dice_coef_loss(y_true, y_pred)
    # Loss 2
    loss2 = loss_mask(y_true, y_pred)
    loss = loss1 + loss2
    return loss


def custom_binary_loss(y_true, y_pred): 
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  # Cancels out when target is 1 
    term_1 = y_true * K.log(y_pred + K.epsilon()) # Cancels out when target is 0
    suma = term_0 + term_1
    return -K.mean(suma, axis=1)+K.std(suma, axis = 1)


def custom_binary_loss_2(y_true, y_pred): 
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    san = K.sum(1-y_true, axis = 0)
    enf = K.sum(y_true, axis = 0)
    dif_sanos = (1 - y_true) * K.abs(y_true-y_pred)  # Cancels out when target is 1 
    dif_sanos = K.sum(dif_sanos, axis = 0)/san
    dif_enf = y_true * K.abs(y_true-y_pred) # Cancels out when target is 0
    dif_enf = K.sum(dif_enf, axis = 0)/enf
    suma = dif_enf + dif_sanos + K.abs(dif_enf-dif_sanos)
    return K.mean(suma)+K.std(suma)