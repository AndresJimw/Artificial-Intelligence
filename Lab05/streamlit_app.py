import os, io, json, random, cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf

BASE = Path(r"D:\Archivos de Usuario\Documents\Artificial-Intelligence\Lab05")
MODEL_PATH  = BASE / "models" / "vgg16_andres_vs_fondo.keras"
LABELS_PATH = BASE / "labels_andres_vs_fondo.json"
DATASET_DIR = BASE / "dataset" / "test"
NEW_IMAGES_DIR = BASE / "new_images"
NEW_IMAGES_DIR.mkdir(exist_ok=True)
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Andrés vs Fondo — VGG-16", layout="centered")
st.title("Andrés vs Fondo — VGG-16")

# CONFIGURAR HILOS CPU
try:
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["OMP_NUM_THREADS"]))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, int(os.environ["OMP_NUM_THREADS"]) // 2))
    st.caption("Uso optimizado de CPU activado.")
except Exception:
    pass  # evita RuntimeError si TF ya se inicializó

# CARGA DE MODELO Y ETIQUETAS
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"No se encontró el modelo en:\n{MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(str(MODEL_PATH))

def load_labels():
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            inv_map = json.load(f)
        labels = [inv_map[str(i)] if str(i) in inv_map else inv_map[i] for i in sorted(map(int, inv_map.keys()))]
        return labels
    return ["Fondo", "Andres"]

model = load_model()
labels = load_labels()

# DETECCIÓN FACIAL (MTCNN)
detector = MTCNN()

def crop_with_margin(img, box, pad=0.25, size=224):
    x, y, w, h = box
    H, W = img.shape[:2]
    cx, cy = x + w/2, y + h/2
    s = int(max(w, h) * (1 + pad))
    x1 = max(0, int(cx - s/2)); y1 = max(0, int(cy - s/2))
    x2 = min(W, int(cx + s/2)); y2 = min(H, int(cy + s/2))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

def detect_and_crop_face(rgb):
    faces = detector.detect_faces(rgb)
    if not faces:
        return None
    f = max(faces, key=lambda d: d["box"][2]*d["box"][3])
    crop = crop_with_margin(rgb, f["box"])
    return crop

# PREDICCIÓN
def preprocess_rgb(rgb):
    x = preprocess_input(rgb.astype(np.float32))
    return np.expand_dims(x, 0)

def predict_face(rgb_face):
    x = preprocess_rgb(rgb_face)
    p = model.predict(x, verbose=0)[0]  # [p_Fondo, p_Andres]
    idx = int(np.argmax(p))
    return labels[idx], float(p[1]), float(p[0])

# INTERFAZ
st.caption("Sube/toma una imagen o selecciona una del conjunto **test**. El modelo detecta y clasifica el rostro.")

mode = st.radio(
    "Modo de prueba",
    ["Subir imagen", "Tomar foto (webcam)", "Probar una del test"],
    horizontal=True
)

img_rgb, img_name = None, None

# Subir imagen 
if mode == "Subir imagen":
    file = st.file_uploader("Elige una imagen (JPG/PNG)", type=["jpg","jpeg","png","bmp","webp"])
    if file:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_rgb = np.array(img)
        img_name = getattr(file, "name", "upload.jpg")

# Tomar foto desde webcam
elif mode == "Tomar foto (webcam)":
    cam = st.camera_input("Toma tu foto")
    if cam:
        img = Image.open(cam).convert("RGB")
        img_rgb = np.array(img)
        img_name = "webcam.jpg"

# Probar una del test
else:
    test_paths = {cls: sorted((DATASET_DIR/cls).glob("*.*")) for cls in ("Andres","Fondo")}
    cls_choice = st.selectbox("Clase del test", ["Andres","Fondo"])
    options = test_paths.get(cls_choice, [])
    if options:
        selected = st.selectbox("Archivo", options, format_func=lambda p: p.name)
        img_rgb = cv2.cvtColor(cv2.imread(str(selected)), cv2.COLOR_BGR2RGB)
        img_name = selected.name

# RESULTADOS
if img_rgb is not None:
    st.image(img_rgb, caption=img_name)

    face = detect_and_crop_face(img_rgb)
    if face is None:
        st.warning("No se detectó ningún rostro en la imagen.")
        label, p_andres, p_fondo = labels[0], 0.0, 1.0
        col1, col2 = st.columns([1,1])
        with col2:
            st.subheader(f"Predicción: {label}")
            st.write(f"p(Andres) = **{p_andres:.3f}**")
            st.write(f"p(Fondo) = **{p_fondo:.3f}**")
            st.progress(min(1.0, max(0.0, p_andres)), text="Probabilidad de 'Andres'")
    else:
        label, p_andres, p_fondo = predict_face(face)

        col1, col2 = st.columns([1,1])
        with col1:
            st.image(face, caption="Rostro detectado")
        with col2:
            st.subheader(f"Predicción: {label}")
            st.write(f"p(Andres) = **{p_andres:.3f}**")
            st.write(f"p(Fondo) = **{p_fondo:.3f}**")
            st.progress(min(1.0, max(0.0, p_andres)), text="Probabilidad de 'Andres'")

    st.markdown("---")

else:
    st.info("Carga, toma o selecciona una imagen para predecir.")
