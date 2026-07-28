import os
import re
import tensorflow as tf
import pandas as pd


# DATAFRAME ---------------------------------------------------------------

def create_dataframe(folder, path = '/home/mr1142/Documents/Data/chest_xray'):
    path = os.path.join('/home/mr1142/Documents/Data/chest_xray', folder)
    for fold in os.listdir(path):
        globals()[fold] = {}
        imgs = os.listdir(os.path.join(path, fold))
        globals()[fold]['path'] = [os.path.join(path, fold)] * len(imgs)
        globals()[fold]['img_name'] = imgs
        globals()[fold]['normal'] = [1 if fold == 'NORMAL' else 0 for _ in range(len(imgs))]
        globals()[fold]['viral'] = [1 if re.search('virus', imgs[i]) else 0 for i in range(len(imgs))]
        globals()[fold]['bacteria'] = [1 if re.search('bacteria', imgs[i]) else 0 for i in range(len(imgs))]
        globals()[fold]['real'] = [0 if fold == 'NORMAL' else 1 if re.search('virus', imgs[i]) else 2 for i in range(len(imgs))]

    for k, v in PNEUMONIA.items():
        v.extend(NORMAL[k])

    df = pd.DataFrame(PNEUMONIA)

    return df


# MODELO ---------------------------------------------------------------

def squeeze_and_excitation(inputs, ratio=16):
    b, _, _, c = inputs.shape
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(c//ratio, activation="relu", use_bias=False)(x)
    x = tf.keras.layers.Dense(c, activation="sigmoid", use_bias=False)(x)
    x = tf.expand_dims(x, 1)
    x = tf.expand_dims(x, 1)
    x = inputs * x
    return x
 

def downsample_block(downsampling_output):
    pix = downsampling_output.shape[2]
    deep = downsampling_output.shape[3]
    name = re.sub(':', '', downsampling_output.name) + '_new'
    x = tf.keras.layers.Conv2D(deep*2,3, padding = 'same', name = name + '_conv')(downsampling_output)
    x = squeeze_and_excitation(x)
    x = tf.keras.layers.MaxPool2D(int(pix/8), name = name + '_max')(x)
    maxpool = tf.keras.layers.GlobalMaxPooling2D()(x)
    return maxpool


def global_max_concat(maxpool_output, previous_layer):
    dense = tf.keras.layers.Dense(128, activation="elu")(maxpool_output)
    unification = tf.keras.layers.concatenate([dense, previous_layer])
    return unification


def modelo(pixels, fine_tune_at = 18, mask = False):
    if mask:
        # model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + 'mask_' + str(pixels) + '.h5'
        model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + str(pixels) + '.h5'
    else:
        model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + str(pixels) + '.h5'
    backbone = tf.keras.models.load_model(model_path)

    inputs = backbone.input

    downsampling_pretrained_output = backbone.layers[18].output
    intermedium = downsample_block(downsampling_pretrained_output)

    dropout_1 = tf.keras.layers.Dropout(0.2, name = "drop_out_1")(intermedium)

    dense_1 = tf.keras.layers.Dense(768, activation="elu")(dropout_1)
    dense_union_1 = global_max_concat(downsample_block(backbone.layers[15].output), dense_1)
    dense_2 = tf.keras.layers.Dense(128, activation="elu")(dense_union_1)
    dense_union_2 = global_max_concat(downsample_block(backbone.layers[11].output), dense_2)
    dense_2 = tf.keras.layers.Dense(128, activation="elu")(dense_union_2)

    dropout_2 = tf.keras.layers.Dropout(0.2, name="dropout_out_2")(dense_2)

    dense_final = tf.keras.layers.Dense(32, activation="elu")(dropout_2)
    outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="fc_out")(dense_final)

    model = tf.keras.Model(inputs, outputs, name="U-Net")

    backbone.trainable = True
    print('\ntrainable variables: {}'.format(len(backbone.trainable_variables)))

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model


def modelo2(pixels, fine_tune_at = 18, mask = False):
    if mask:
        # model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + 'mask_' + str(pixels) + '.h5'
        model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + str(pixels) + '.h5'
    else:
        model_path = '/home/mr1142/Documents/Data/models/unsupervised_' + str(pixels) + '.h5'
    
    backbone = tf.keras.models.load_model(model_path)

    inputs = backbone.input

    downsampling_pretrained_output = backbone.layers[18].output
    maxpool_intermedium = tf.keras.layers.GlobalMaxPooling2D()(downsampling_pretrained_output)

    dropout_1 = tf.keras.layers.Dropout(0.2, name = "drop_out_1")(maxpool_intermedium)

    dense_1 = tf.keras.layers.Dense(768, activation="elu")(dropout_1)
    dense_union_1 = global_max_concat(tf.keras.layers.GlobalMaxPooling2D()(backbone.layers[15].output), dense_1)
    dense_2 = tf.keras.layers.Dense(128, activation="elu")(dense_union_1)
    dense_union_2 = global_max_concat(tf.keras.layers.GlobalMaxPooling2D()(backbone.layers[11].output), dense_2)
    dense_2 = tf.keras.layers.Dense(128, activation="elu")(dense_union_2)

    dropout_2 = tf.keras.layers.Dropout(0.2, name="dropout_out_2")(dense_2)

    dense_final = tf.keras.layers.Dense(32, activation="elu")(dropout_2)
    outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="fc_out")(dense_final)

    model = tf.keras.Model(inputs, outputs, name="U-Net")

    backbone.trainable = True
    print('\ntrainable variables: {}'.format(len(backbone.trainable_variables)))

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model

