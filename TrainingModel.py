import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_centro_datos.h5')

# Cargar una imagen de prueba y preprocesarla
def cargar_y_preprocesar_imagen(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=(224, 224))  # Redimensionar a 224x224
    img_array = image.img_to_array(img)  # Convertir a array
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para que sea compatible con el modelo
    img_array /= 255.0  # Normalizar los valores de los píxeles
    return img_array

# Realizar la predicción
def predecir_imagen(modelo, ruta_imagen, class_labels):
    img_array = cargar_y_preprocesar_imagen(ruta_imagen)
    prediccion = modelo.predict(img_array)
    clase_predicha = np.argmax(prediccion, axis=1)
    return class_labels[clase_predicha[0]], prediccion[0]

# Etiquetas de las clases (deberías ajustarlas según tu dataset)
class_labels = list(train_generator.class_indices.keys())

# Ruta de la imagen de prueba
ruta_imagen = 'prueba.jpg'

# Predecir la clase de la imagen
clase_predicha, probabilidades = predecir_imagen(model, ruta_imagen, class_labels)

# Mostrar el resultado
print(f"Predicción: {clase_predicha}")
print(f"Probabilidades por clase: {probabilidades}")

# Mostrar la imagen con la clase predicha
img = image.load_img(ruta_imagen)
plt.imshow(img)
plt.title(f"Predicción: {clase_predicha}")
plt.show()
