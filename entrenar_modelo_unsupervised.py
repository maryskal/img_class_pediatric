import argparse
import os
import pandas as pd
import numpy as np
import funciones_modelos.logs as logs
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model")          
    parser.add_argument('-m',
                        '--mask',
                        action = 'store_true',
                        help="use mask or not")     
    parser.add_argument('-px',
                        '--pixels',
                        type=int,
                        default=256,
                        help="pixels for the img")
    parser.add_argument('-b',
                        '--batch',
                        type=int,
                        default=8,
                        help="batch size")                


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    name = args.name
    mask = args.mask
    path = '/home/mr1142/Documents/Data/NIH'
    batch = args.batch
    epoch = 200
    pixels = args.pixels

    #----------------------------------------------------
    import funciones_unsupervised.unet_funct as u_net

    metrics = ['mean_squared_error', 'mean_absolute_error']

    def unet():
        unet_model = u_net.build_unet_model(pixels)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss='mean_squared_error',
                        metrics=metrics)
        return unet_model
    #----------------------------------------------------

    df = pd.read_csv(os.path.join(path, 'Data_Entry_2017.csv'))
    names = list(df['Image Index'])

    np.random.shuffle(names)
    names_train = names[:round(len(names)*0.8)]
    names_test = names[round(len(names)*0.8):]

    from funciones_unsupervised.data_generator import DataGenerator as gen

    traingen = gen(names_train, batch, pixels, mask)
    testgen = gen(names_test, batch, pixels, mask)

    callb = [logs.tensorboard(name), logs.early_stop(3)]
    unet_model = unet()
    unet_model.summary()

    history = unet_model.fit(traingen, 
                            validation_data = testgen,
                            batch_size = batch,
                            epochs = 200,
                            callbacks= callb,
                            shuffle = True) 

    min = min(history.history['val_mean_squared_error'])
    unet_model.save('/home/mr1142/Documents/Data/models/' +  name + '_' + round(min[2:],4) + '.h5')
