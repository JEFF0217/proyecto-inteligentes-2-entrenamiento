# from os import name
import time
import tensorflow as tf
import tensorflow_addons as tfa
import keras
import numpy as np
import cv2
from sklearn.model_selection import KFold
import os
# Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, MaxPool2D, Reshape, Dense, Flatten

CATEGORIAS = ["beachballs", "billiardball", "bowlingball", "football", "golfball",
              "paintballs", "pokemonballs", "soccerball", "tennisball", "volleyball"]


def cargarDatos(rutaOrigen, numeroCategorias, ancho, alto):
    imagenesCargadas = []
    valorEsperado = []
    for categoria in range(1, len(numeroCategorias)):
        dir = rutaOrigen + CATEGORIAS[categoria]
        files = os.listdir(dir)

        for file_name in files:
            ruta = dir + "/"+file_name
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto),
                                interpolation=cv2.INTER_AREA)
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(len(numeroCategorias))
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados


#################Implementación del modelo ####################
# Definir dimension imagen
width = 128
height = 128
pixeles = width * height
num_channels = 1  # Si imagen blanco/negro = 1     rgb = 3
img_shape = (width, height, num_channels)
# cantidad elementos clasificar
#Cambiar ruta para buscar archivos
dirc = "D:/universidad/2022-2/inteligentes 2/archive/"

# CargaImagen
imagenes, probabilidades = cargarDatos(
    dirc+"train/", CATEGORIAS, width, height)
imagenesPrueba, probabilidadesPrueba = cargarDatos(
    dirc+"test/", CATEGORIAS, width, height)


model = Sequential()
# Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))
# Reformar imagen
model.add(Reshape(img_shape))

# Capas convolucionales
model.add(Conv2D(kernel_size=5, strides=1, filters=16,
          padding='same', activation='relu', name='capa_convolucion_1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=1, filters=36,
          padding='same', activation='relu', name='capa_convolucion_2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Aplanar imagen
model.add(Flatten())
# Capa densa
model.add(Dense(128, activation='relu'))


# Capa salida
model.add(Dense(len(CATEGORIAS), activation='softmax'))


# COMPILACIÓN
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=['accuracy'])

# CROSS-VALIDATION
numero_fold = 1
accuracy_fold = []
loss_fold = []

kFold = KFold(n_splits=5, shuffle=True)

# JUNTAMOS LOS DATOS PARA QUE LA VALIDACIÓN CRUZADA LOS ORDENE
X = np.concatenate((imagenes, imagenesPrueba), axis=0)
y = np.concatenate((probabilidades, probabilidadesPrueba), axis=0)


# Tiempo de inicio de ejecución.
inicio = time.time()

for train, test in kFold.split(X, y):
    print("##################Training fold ", numero_fold,
          "###################################")
    model.fit(X[train], y[train],
              epochs=15,  # Epocas--> Cantidad de veces que debe repetir el entrenamiento
              batch_size=191  # Batch --> Cantidad de datos que puede cargar en memoria para realizar el entrenamiento en una fase
              )
    metricas = model.evaluate(X[test], y[test])
    accuracy_fold.append(metricas[1])
    loss_fold.append(metricas[0])
    numero_fold += 1


# Tiempo de fin de ejecución.
fin = time.time()
# Tiempo de ejecución.
tiempo_total = fin - inicio
print(tiempo_total, "tiempo total")

for i in range(0, len(loss_fold)):
    print("Fold ", (i+1), "- Loss(Error)=",
          loss_fold[i], " - Accuracy=", accuracy_fold[i])
print("-------Average scores-------")
print("Loss", np.mean(loss_fold))
print("Accuracy", np.mean(accuracy_fold))

# Guardar el modelo
ruta = "models/modelo_1v5.h5"
model.save(ruta)

# Resumen - Estructura de la red
model.summary()
