import os, cv2
from pathlib import Path
from mtcnn.mtcnn import MTCNN

BASE = Path(__file__).resolve().parents[1]
SRC  = BASE / "lookalikes_raw"
DST  = BASE / "lookalikes_crops_224"
DST.mkdir(parents=True, exist_ok=True)

det = MTCNN()
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

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

if not SRC.exists():
    raise FileNotFoundError(f"No existe {SRC}. Revisa tu estructura de carpetas.")

total_saved = 0

persons = sorted([d for d in os.listdir(SRC) if (SRC / d).is_dir()])

for person in persons:
    src_dir = SRC / person
    dst_dir = DST / person
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(EXTS)]
    print(f"[INFO] {person}: {len(files)} archivos encontrados en {src_dir}")

    saved = 0
    for name in files:
        path = src_dir / name
        img  = cv2.imread(str(path))
        if img is None:
            continue

        faces = det.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not faces:
            # sin detección, salta
            continue

        # si hay varias caras, usa la más grande
        f = max(faces, key=lambda d: d["box"][2] * d["box"][3])
        crop = crop_with_margin(img, f["box"])
        if crop is None:
            continue

        out = dst_dir / name
        cv2.imwrite(str(out), crop)
        saved += 1

    print(f"[OK] {person}: recortadas {saved} → {dst_dir}")
    total_saved += saved

print(f"[DONE] Total de recortes guardados: {total_saved} en {DST}")
