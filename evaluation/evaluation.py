import pandas as pd
import os

def save_training(name, otros_datos, history):
    datos = history.history
    max_values = []
    for v in datos.values():
        max_values.append(max(v))
    name = name + '_auc-' + str(max(datos['val_auc']))[2:4]
    # Save maximos
    path = '/home/mr1142/Documents/Data/models/neumonia_pediatric/training_data/train_max_pediatric.csv'
    save(name, otros_datos, max_values, path)
    # Save all train
    path = '/home/mr1142/Documents/Data/models/neumonia_pediatric/training_data'
    pd.DataFrame(datos).to_csv(os.path.join(path, name + '_data.csv'), index = False)
    return name


def save(name, otros_datos, results, path):
    df = pd.read_csv(path)
    save = [name] + otros_datos + results
    try:
        # Si ya existe el modelo, se sobreescriben las métricas
        i = df[df['nombre'] == name].index
        df.loc[i[0]] = save
    except:
        df.loc[len(df.index)] = save
    df.reset_index(drop=True)
    df.to_csv(path, index = False)
