import os, platform
import cv2, time, threading
from ultralytics import YOLO
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# 1) Intentar ruta fija de Windows
DEFAULT_WIN_PATH = r"D:\Archivos de Usuario\Documents\Artificial-Intelligence\runs\detect\custom_yolo_model\weights\best.pt"

# 2) Si hay variable de entorno WEIGHTS, úsala
env_weights = os.getenv("WEIGHTS")
if env_weights and Path(env_weights).expanduser().exists():
    WEIGHTS = str(Path(env_weights).expanduser())
# 3) Si no, usa la ruta por defecto
elif Path(DEFAULT_WIN_PATH).exists():
    WEIGHTS = DEFAULT_WIN_PATH
else:
    # 4) Búsqueda automática relativa al script o al cwd (Linux/servidor)
    candidates = []
    here = Path(__file__).resolve().parent
    rel = Path("runs/detect/custom_yolo_model/weights/best.pt")
    # Probar en:
    for base in {here, here.parent, Path.cwd()}:
        p = (base / rel).resolve()
        candidates.append(p)
        if p.exists():
            WEIGHTS = str(p)
            break
    else:
        raise FileNotFoundError(
            "No encuentro el modelo 'best.pt'. Define WEIGHTS o coloca el peso en:\n"
            + "\n".join(f"- {c}" for c in candidates)
        )

# Comprobación final
if not Path(WEIGHTS).exists():
    raise FileNotFoundError(f"No encuentro el modelo en:\n{WEIGHTS}\nVerifica la ruta o define WEIGHTS.")

CAM_INDEX = 0  # prueba 1 si 0 no abre

# === HIPERPARÁMETROS ===
CONF, IOU = 0.03, 0.60
IMGSZ, MAX_DET = 640, 40
DEVICE = "cpu"  # en Jetson con GPU: "cuda"

# === MODELO Y CLASE ACTIVA ===
model = YOLO(WEIGHTS)
name_to_id = {str(v).lower(): k for k, v in model.names.items()}
if "pen" not in name_to_id:
    raise RuntimeError(f'La clase "pen" no existe en model.names: {model.names}')
PEN_ID = name_to_id["pen"]
ACTIVE_IDS = [PEN_ID]

# === SHARED BUFFER PARA STREAM ===
latest_jpeg = None
lock = threading.Lock()
running = True

def is_jetson():
    return os.path.exists("/etc/nv_tegra_release")

def list_video_devices():
    base = Path("/dev")
    if not base.exists():
        return []
    return sorted([int(p.name.replace("video",""))
                   for p in base.glob("video*")
                   if p.name.replace("video","").isdigit()])

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720,
                       display_width=640, display_height=480, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def open_camera_linux_or_jetson(preferred_index=0, width=640, height=480):
    indices = list_video_devices()
    candidates = [preferred_index] + [i for i in indices if i != preferred_index] if indices else [preferred_index]
    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            print(f"[INFO] Cámara V4L2 abierta en /dev/video{idx}")
            return cap

    if is_jetson():
        for sid in (0, 1):  # pruebo hasta dos sensores comunes
            pipe = gstreamer_pipeline(sensor_id=sid, display_width=width, display_height=height, framerate=30)
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"[INFO] Cámara CSI abierta con GStreamer (sensor-id={sid})")
                return cap

    return None

def capture_and_detect():
    global latest_jpeg, running
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = open_camera_linux_or_jetson(preferred_index=CAM_INDEX, width=640, height=480)

    if cap is None or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba CAM_INDEX=1 o conecta una cámara USB. "
                           "En Jetson, asegúrate de tener una cámara CSI/USB compatible y GStreamer habilitado.")

    last_t, fps = time.time(), 0.0
    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                break

            res = model(
                frame, conf=CONF, iou=IOU, classes=ACTIVE_IDS,
                imgsz=IMGSZ, max_det=MAX_DET, agnostic_nms=False,
                device=DEVICE, verbose=False
            )[0]

            out = res.plot()

            pens = 0
            if res.boxes is not None and len(res.boxes):
                cls_np = res.boxes.cls.int().cpu().numpy().tolist()
                pens = cls_np.count(PEN_ID)

            now = time.time(); dt = now - last_t
            if dt > 0: fps = 0.9*fps + 0.1*(1.0/dt)
            last_t = now

            cv2.putText(out, f"pen: {pens}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(out, f"conf>={CONF:.2f}  iou={IOU:.2f}  fps~{fps:.1f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with lock:
                    latest_jpeg = buf.tobytes()
    finally:
        cap.release()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!doctype html>
<title>YOLO pen</title>
<body style="margin:0;background:#111;">
<div style="color:#eee;font:16px sans-serif;padding:8px;">YOLO pen (Ctrl+C para salir)</div>
<img src="/stream" style="max-width:100%;display:block;margin:0 auto;"/>
</body>""")
            return
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while running:
                    with lock:
                        frame = latest_jpeg
                    if frame is None:
                        time.sleep(0.01); continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame + b"\r\n")
                    time.sleep(0.001)
            except BrokenPipeError:
                pass
            return
        self.send_error(404)

def run_server(host="0.0.0.0", port=8000):
    httpd = HTTPServer((host, port), MJPEGHandler)
    print(f"[INFO] Abre http://127.0.0.1:{port} para ver el stream.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    t = threading.Thread(target=capture_and_detect, daemon=True)
    t.start()
    try:
        run_server()
    finally:
        running = False
        t.join(timeout=2.0)
        print("[INFO] Finalizado.")
