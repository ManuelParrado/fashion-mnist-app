import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model('fashion_mnist.keras')

# Crear la interfaz de usuario
st.title('Clasificación Fashion MNIST')
st.write('Sube una imagen para clasificarla como una categoria de ropa.')

uploaded_file = st.file_uploader('Sube una imagen en escala de grises de 28x28 píxeles.', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Procesar la imagen
    image = Image.open(uploaded_file).convert('L') # Convertir RGB a Blanco y Negro
    image = image.resize((28, 28)) # Redimensionar a 28x28 píxeles
    img_array = np.array(image) / 255.0 # Normalizar
    # El primer 1 indica que sólo hay una imagen, luegos las dimensiones
    # y el último 1 indica que sólo hay un canal de color.
    img_array = img_array.reshape(1, 28, 28, 1)

    # Mostrar la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Predicción 
    prediction = model.predict(img_array)
    classes = ['Camiseta', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']
    st.write('Predicción', classes[np.argmax(prediction)])
