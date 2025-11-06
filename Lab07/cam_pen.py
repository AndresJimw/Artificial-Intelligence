import cv2, time, threading
from ultralytics import YOLO
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# === RUTAS ===
WEIGHTS = r"D:\Archivos de Usuario\Documents\Artificial-Intelligence\runs\detect\custom_yolo_model\weights\best.pt"
if not Path(WEIGHTS).exists():
    raise FileNotFoundError(f"No encuentro el modelo en:\n{WEIGHTS}\nVerifica la ruta.")

CAM_INDEX = 0  # prueba 1 si 0 no abre

# === HIPERPARÁMETROS ===
CONF, IOU = 0.02, 0.60
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

def capture_and_detect():
    global latest_jpeg, running
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba CAM_INDEX=1.")

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
