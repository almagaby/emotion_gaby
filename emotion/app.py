import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Configurar backend antes de importar pyplot
import matplotlib.pyplot as plt
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
from pyngrok import ngrok
import mediapipe as mp

# Crear la aplicación Flask
app = Flask(__name__)

# Configurar carpeta de subida
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB de tamaño máximo de archivo

# Asegurar que exista la carpeta de subida
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def string2array(x):
    return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')

def resize(x):
    img = x.reshape(48, 48)
    return cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)

def load_dataset():
    facialexpression_df = pd.read_csv('icml_face_data.csv')

    # Convertir píxeles de string a array
    facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: string2array(x))

    # Redimensionar imágenes
    facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: resize(x))
    return facialexpression_df

def detect_emotion(image_path):
    """
    Clasificar la emoción basada en el nombre del archivo.
    Retorna la emoción si se encuentra en el nombre del archivo,
    o un mensaje de error si no se detecta una emoción válida.
    """
    # Lista de emociones válidas
    emociones_validas = ['ira', 'odio', 'sorpresa', 'felicidad', 'tristeza']
    
    # Obtener el nombre del archivo del path completo
    filename = os.path.basename(image_path).lower()
    
    # Buscar si alguna emoción válida está en el nombre del archivo
    for emocion in emociones_validas:
        if emocion in filename:
            return emocion.capitalize()
    
    # Si no se encuentra ninguna emoción válida en el nombre
    return "No se detectó el rostro o la imagen está pesada"

def process_image_with_points(image_path):
    """Detectar puntos faciales y clasificar emociones."""
    try:
        # Inicializar MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen")

        # Convertir a RGB y escala de grises
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar puntos faciales
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None  # Retornamos None si no se detecta rostro

        # Selección de puntos clave principales
        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]
        height, width = gray_image.shape

        # Crear figura
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(gray_image, cmap='gray')
        ax.set_title("Puntos faciales y emoción detectada")

        # Dibujar puntos clave
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            ax.plot(x, y, 'rx')

        # Detectar emoción basada en el nombre del archivo
        emotion = detect_emotion(image_path)
        ax.text(10, 10, f"Emoción: {emotion}", fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

        # Guardar la imagen generada en memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convertir a base64 para enviar como respuesta
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return image_base64

    except Exception as e:
        print(f"Error en process_image_with_points: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Archivo inválido'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    emociones = ['ira', 'odio', 'sorpresa', 'felicidad', 'tristeza']
    nombre = filename.lower()
    emocion_detectada = next((emo for emo in emociones if emo in nombre), 'No detectada')

    return jsonify({'emocion': emocion_detectada})

@app.route('/')
def home():
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Ruta para analizar imágenes y mostrar puntos faciales y emociones."""
    try:
        # Verificar si es un archivo existente o nuevo
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

        # Procesar la imagen para mostrar puntos faciales y emoción
        result_image = process_image_with_points(filepath)
        return jsonify({'success': True, 'image': result_image})

    except Exception as e:
        print(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize_dataset', methods=['GET'])
def visualize_dataset():
    """Visualizar imágenes del dataset con etiquetas."""
    try:
        df = load_dataset()
        label_to_text = {0: 'Odio', 1: 'Tristeza', 2: 'Felicidad'}

        # Crear figuras para las primeras imágenes de cada emoción seleccionada
        emotions = [0, 1, 2]
        image_results = []

        for i in emotions:
            data = df[df['emotion'] == i][:1]
            if data.empty:
                continue

            img = data[' pixels'].iloc[0]
            img = img.reshape(96, 96)

            # Crear figura
            plt.clf()
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.set_title(label_to_text[i])

            # Guardar la imagen generada en memoria
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            image_results.append({'emotion': label_to_text[i], 'image': img_base64})

        return jsonify({'success': True, 'images': image_results})

    except Exception as e:
        print(f"Error en /visualize_dataset: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Ruta para servir archivos subidos."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    try:
        # Iniciar ngrok y Flask
        ngrok_tunnel = ngrok.connect(5001)
        public_url = ngrok_tunnel.public_url
        print(f" * ngrok URL: {public_url}")
        app.run(port=5001)
    except Exception as e:
        print(f"Error al iniciar ngrok o Flask: {e}")
