import os, cv2
from mtcnn.mtcnn import MTCNN

SRC = r".\frames_andres"
DST = r".\andres_crops_224"
os.makedirs(DST, exist_ok=True)

det = MTCNN()

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

saved = 0
files = [f for f in os.listdir(SRC) if f.lower().endswith((".jpg",".jpeg",".png"))]
for name in files:
    path = os.path.join(SRC, name)
    img  = cv2.imread(path)
    if img is None: 
        continue

    faces = det.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(faces) == 0:
        # no se detectó cara, lo ignoramos
        continue

    # si hay varias, toma la más grande
    f = max(faces, key=lambda d: d["box"][2]*d["box"][3])
    crop = crop_with_margin(img, f["box"])
    if crop is None:
        continue

    out = os.path.join(DST, f"andres_{name}")
    cv2.imwrite(out, crop)
    saved += 1

print(f"[OK] Caras recortadas y guardadas: {saved} en {DST}")
