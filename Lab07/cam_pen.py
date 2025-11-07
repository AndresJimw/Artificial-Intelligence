import os, platform, time
from pathlib import Path
import cv2
from ultralytics import YOLO

DEFAULT_WIN_PATH = r"D:\Archivos de Usuario\Documents\Artificial-Intelligence\runs\detect\custom_yolo_model\weights\best.pt"

env_weights = os.getenv("WEIGHTS")
if env_weights and Path(env_weights).expanduser().exists():
    WEIGHTS = str(Path(env_weights).expanduser())
elif Path(DEFAULT_WIN_PATH).exists():
    WEIGHTS = DEFAULT_WIN_PATH
else:
    candidates = []
    here = Path(__file__).resolve().parent
    rel = Path("runs/detect/custom_yolo_model/weights/best.pt")
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

if not Path(WEIGHTS).exists():
    raise FileNotFoundError(f"No encuentro el modelo en:\n{WEIGHTS}\nVerifica la ruta o define WEIGHTS.")

CAM_INDEX = 0
CONF, IOU = 0.03, 0.60
IMGSZ, MAX_DET = 640, 40
DEVICE = "cpu"  # en Jetson con GPU: "cuda"

model = YOLO(WEIGHTS)
name_to_id = {str(v).lower(): k for k, v in model.names.items()}
if "pen" not in name_to_id:
    raise RuntimeError(f'La clase "pen" no existe en model.names: {model.names}')
PEN_ID = name_to_id["pen"]
ACTIVE_IDS = [PEN_ID]

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

def open_camera(preferred_index=0, width=640, height=480):
    if platform.system() == "Windows":
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            for idx in (preferred_index, 1):
                cap = cv2.VideoCapture(idx, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if cap.isOpened():
                    return cap
        return None
    indices = list_video_devices()
    candidates = [preferred_index] + [i for i in indices if i != preferred_index] if indices else [preferred_index]
    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            return cap
    if is_jetson():
        for sid in (0, 1):
            pipe = gstreamer_pipeline(sensor_id=sid, display_width=width, display_height=height, framerate=30)
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap
    cap = cv2.VideoCapture(preferred_index, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap if cap.isOpened() else None

def ensure_gui_available():
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        return True
    except cv2.error:
        return False

def main():
    if not ensure_gui_available():
        raise RuntimeError("Tu OpenCV es 'headless'. Instala opencv-python (no headless) para usar ventanas.")
    cap = open_camera(preferred_index=CAM_INDEX, width=640, height=480)
    if cap is None or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba CAM_INDEX=1 o conecta una cámara USB.")
    cv2.namedWindow("YOLO pen", cv2.WINDOW_NORMAL)
    last_t, fps, drop_reads = time.time(), 0.0, 0
    try:
        while True:
            ok, img = cap.read()
            if not ok or img is None:
                drop_reads += 1
                if drop_reads > 10:
                    cap.release()
                    cap = open_camera(preferred_index=CAM_INDEX, width=640, height=480)
                    if cap is None or not cap.isOpened():
                        break
                    drop_reads = 0
                continue
            results = model(img, stream=True, conf=CONF, iou=IOU, classes=ACTIVE_IDS,
                            imgsz=IMGSZ, max_det=MAX_DET, device=DEVICE, verbose=False)
            pens = 0
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    if cls == PEN_ID:
                        pens += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    cv2.putText(img, "pen", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
            now = time.time(); dt = now - last_t
            if dt > 0: fps = 0.9*fps + 0.1*(1.0/dt)
            last_t = now
            cv2.putText(img, f"pen: {pens}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f"conf>={CONF:.2f}  iou={IOU:.2f}  fps~{fps:.1f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("YOLO pen", img)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
    finally:
        try: cap.release()
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
