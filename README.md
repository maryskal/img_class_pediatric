# DATOS
## Dataset completo
El dataset completo es originario de: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Cuenta con las siguientes carpetas:
- train
    - normal: 1341 images
    - pneumonia: 3875 images
- test
    - normal: 234
    - pneumonia: 390
- val
    - normal: 8
    - pneumonia: 8
- chest_xray
    - train
        - normal: 1342
        - pneumonia: 1876
    - test
        - normal: 234
        - pneumonia: 390
    - val
        - normal: 9
        - pneumonia: 9

Se ha utilizado la carpeta train para entrenar y la test con la val se utilizarán para testear

Los resultados del entrenamiento de estos modelos se han guardado en: 
- /Documents/Data/models/neumonia/training_data/train_max_unsupervised.csv

Los resultados del test de los modelos entrenados de esta manera se guardaron en un dataframe:
- /Documents/Data/models/neumonia/validation_results/image_class_evaluation_unsupervised.csv


# PREPROCESADO
En algunos casos se aplicó un modelo que enmascara el tórax antes de todo el preprocesado (***funciones_imagenes/mask_function.py***):
- Paso a escala de grises
- Resize a 256,256
- Aplicación del modelo y extracción de la máscara
- Resize de la máscara al tamaño de la imagen original
- Quitar agujeros y labels extra a la mascara
- Aplicar la máscara sobre la imagen original
- Desnormalizar el resultado

Además a todas las imágenes se les aplicó (funciones_imagenes/prepare_img_fun.py):
- Paso a escala de grises: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) -> (pix,pix)
- Resize: img = cv2.resize(img, (512,512)) -> (512,512)
- Expandir dimension: img = np.expand_dims(img, axis=-1)
- Clahe: con clip limit sin especificar o de 2
- Normalización con z-score: (img - np.mean(img))/ np.std(img)


# MODELOS
Primero se construyó una U-Net que tenía como input lo mismo que como output, la idea era conseguir
sintetizar toda la información de una Rx de tórax en el espacio de parámetros al fondo de la U-Net.
Esta red se entrenó con todas las imágenes del NIH:
    - Con mascara o sin máscara
    - Con 512 o 256 pixels

Los modelos producto de estos entrenamientos se utilizaron como backbone para el siguiente modelo.

El segundo modelo utilizó como backbone el downsampling del primero, que había simplificado la información
de la imagen, así se pretende no generar overfiting con las pocas imágenes que tenemos.

Se aplicaron dos arquitecturas diferentes:
1 - A la salida de este backbone se aplicó un método de atención (Channel Attention and Squeeze-and-Excitation Networks)
para seleccionar los canales con mayor relevancia (output -> conv2D -> attention -> maxpool -> globalMaxPooling -> dense(128 chanels)). 
También se extrajo el outcome de la capa 11 y 15 y se aplicó el mismo esquema, antes de concatenarla con diferentes capas densas.

2 - Los tres outputs del backbone mencionados en la previa se concatenaron simplemente mediante GlobalMaxPooling, sin aplicar métodos
de atención 

## Hiperparámetros
Los hiperparámetros que se han mantenido fijos han sido
- batch size de 8
- train - test proportion de 0.8-0.2
- Optimizador Adam

Los hiperparámetros que se han ido variando han sido
- pixels (que a su vez modificaban el backbone)
- modelo (1 o 2)
- mascara (que a su vez modificaba el backbone)
- frozen layer
- learning rate
- loss
- augmentation

## Modelos utilizados
Para entrenar con máscara se ha necesitado utilizar el modelo unet_final_renacimiento_validation_6.h5.


# ENTRENAMIENTOS
## Entrenamientos independientes
Se han hecho un par de entrenamientos independientes, con clahe no definido.

## Hyperparameter tunning
El resto de entrenamientos se han realizado con mango, algunos con clahe no definido y otros con clahe de 20.
Los resultados se han guardado en:
- /Documents/Data/models/neumonia/training_data/train_max_unsupervised.csv
Se ha testado con model.evaluate y los resultados se han guardado en:
- /Documents/Data/models/neumonia/validation_results/image_class_evaluation_unsupervised.csv
