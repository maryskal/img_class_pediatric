import os
import cv2
import numpy as np
import otras_funciones.logs as logs
import tensorflow as tf
from sklearn.model_selection import train_test_split


def train(modelo, frozen_layer, lr, pixels, loss, mask, augment):
    batch = 8
    epoch = 200
    name = modelo + '_' + str(frozen_layer) + '_' + str(lr)[2:] + '_' + str(pixels) + '_' + loss + '_' + str(mask)[:1] + str(augment)[:1]
    #----------------------------------------------------
    import funciones_imagenes.pediatric_model as fine
    import funciones_imagenes.data_generator as gen
    import funciones_imagenes.losses as losses
    
    # DATA ----------------------------------------------------
    df = fine.create_dataframe('train')
    train, test = train_test_split(df, test_size=0.2, stratify = df.real)

    # Agumentation, pixels and mask
    if augment:
        traingen = gen.DataGenerator_augment(train, 8, pixels, mask)
        testgen = gen.DataGenerator_augment(test, 8, pixels, mask)
    else:
        traingen = gen.DataGenerator(train, 8, pixels, mask)
        testgen = gen.DataGenerator(test, 8, pixels, mask)

    # MODELO ----------------------------------------------------
    # Type of model, pixels and frozen layer
    if modelo == 'model_1':
        model = fine.modelo(pixels, frozen_layer)
    else:
        model = fine.modelo2(pixels, frozen_layer)

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
    callb = [logs.early_stop(5)]

    # TRAIN ----------------------------------------------------
    history = model.fit(traingen, 
                        validation_data = testgen,
                        batch_size = batch,
                        callbacks = callb,
                        epochs = epoch,
                        shuffle = True)
    
    print('numero de errores: %i' % traingen.errors)
    print('error location: {}'.format(traingen.errors_location))
    
       
    # EVALUACIÓN ----------------------------------------------------
    datos = [modelo, pixels, mask, augment, frozen_layer, loss, lr, batch]
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
    metricas['auc_mean']= (metricas['auc_0']+metricas['auc_1']+metricas['auc_2'])/3

    return metricas['auc_mean']