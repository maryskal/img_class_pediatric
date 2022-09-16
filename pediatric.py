import argparse
import os
import otras_funciones.logs as logs
import tensorflow as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
lr = 1e-4
name = 'prueba_2'
fzl = 18
mask = False
pixels = 512
augment = True
modelo = 'model_2'
loss = 'binary_crossentropy'


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
    parser.add_argument('-lr',
                        '--lr',
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument('-m',
                        '--mask',
                        action = 'store_true',
                        help="apply mask")
    parser.add_argument('-fz',
                        '--frozen_layer',
                        type=int,
                        default=18,
                        help="frozen layer")       
    parser.add_argument('-px',
                        '--pixels',
                        type=int,
                        default=512,
                        help="pixels")
    parser.add_argument('-lo',
                        '--loss',
                        type=str,
                        default='binary_crossentropy',
                        help="loss")
    parser.add_argument('-mo',
                        '--model',
                        type=str,
                        default='model_1',
                        help="model")
    parser.add_argument('-aug',
                        '--augment',
                        action = 'store_true',
                        help="augmentation")  

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    name = args.name
    lr = args.lr
    mask = args.mask
    fzl = args.frozen_layer
    pixels = args.pixels
    loss = args.loss
    modelo = args.model
    augment = args.augment
    batch = 8
    epoch = 200

    #----------------------------------------------------
    import funciones_imagenes.pediatric_model as fine
    import funciones_imagenes.data_generator as gen
    import funciones_imagenes.losses as losses

    # DATA ----------------------------------------------------
    df = fine.create_dataframe('train')
    train, test = train_test_split(df, test_size=0.2, stratify = df.real)

    # Agumentation
    if augment:
        batch = int(batch/2)
        traingen = gen.DataGenerator_augment(train, batch, pixels, mask)
        testgen = gen.DataGenerator_augment(test, batch, pixels, mask)
    else:
        traingen = gen.DataGenerator(train, batch, pixels, mask)
        testgen = gen.DataGenerator(test, batch, pixels, mask)

    # MODELO ----------------------------------------------------
    # Model selection, pixels, mask and frozen layer
    if modelo == 'model_1':
        model = fine.modelo(pixels, fzl, mask)
    else:
        model = fine.modelo2(pixels, fzl, mask)

    # COMPILADO ----------------------------------------------------
    # Loss
    if loss == 'custom_loss':
        ls = losses.custom_binary_loss
    else:
        ls = 'binary_crossentropy'

    # Compilado
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                    loss = ls,
                    metrics = ['binary_accuracy', 'Precision', 'AUC'])

    # CALLBACK ----------------------------------------------------
    callb = [logs.tensorboard(name), logs.early_stop(5)]

    # TRAIN ----------------------------------------------------
    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch)
    
    print('numero de errores: %i' % traingen.errors)
    print('error location: {}'.format(traingen.errors_location))

    datos = [modelo, pixels, mask, augment, fzl, loss, lr, batch]

    # EVALUACIÓN ----------------------------------------------------
    # SAVE TRAINING
    import funciones_evaluacion.evaluation as ev
    # Guardar el train
    name = ev.save_training(name, 
            datos, history)
    print('TRAINING GUARDADO')

    # SAVE MODEL
    model.save('/home/mr1142/Documents/Data/models/neumonia_pediatric/' + name + '.h5')
    print('MODELO GUARDADO')

    # EVALUATION 
    results = model.evaluate(testgen, batch_size=batch)
    p = '/home/mr1142/Documents/Data/models/neumonia_pediatric/validation_results/image_class_evaluation_pediatric.csv'
    ev.save(name, datos, results, p)
    print('EVALUATE GUARDADO')

    # PREDICTION
    import funciones_evaluacion.metrics_and_plots as met
    import funciones_imagenes.prepare_img_fun as fu

    # Predecimos
    y_real = np.array(test[['normal', 'viral', 'bacteria']])
    y_pred = np.zeros((len(test), 3))
    for i in range(len(test)):
        img = cv2.imread(os.path.join(test['path'].iloc[i], test.img_name.iloc[i]))
        y_pred[i,...] = model.predict(np.expand_dims(fu.get_prepared_img(img, pixels, mask, clahe_bool=True), 0))

    # Calculo métricas y guardo
    metricas, _ = met.metricas_dict(y_real, y_pred)
    p = '/home/mr1142/Documents/Data/models/neumonia_pediatric/validation_results/prediction_validation_metrics.csv'
    ev.save(name, datos, list(metricas.values()), p)
