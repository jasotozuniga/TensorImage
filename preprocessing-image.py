from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Generador de imágenes con aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Normalización de píxeles
    shear_range=0.2,                # Aplicar cortes a la imagen
    zoom_range=0.2,                 # Aplicar zoom
    horizontal_flip=True,           # Volteo horizontal
    validation_split=0.2            # Dividir el dataset en entrenamiento y validación
)

# Directorio del dataset
dataset_dir = "dataset/"

# Preparar las imágenes para el entrenamiento
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),         # Redimensionar imágenes
    batch_size=32,
    class_mode='categorical',       # Clasificación multiclase
    subset='training'               # Subconjunto de entrenamiento
)

# Preparar las imágenes para la validación
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'             # Subconjunto de validación
)
