import os, cv2, random, shutil, math, re
import numpy as np
from glob import glob
from tqdm import tqdm

BASE = r"D:\Archivos de Usuario\Documents\Artificial-Intelligence\Lab05"
SRC_ANDRES = os.path.join(BASE, "andres_crops_224")
RAW_BG     = os.path.join(BASE, "external", "backgrounds_raw", "101_ObjectCategories")
RAW_LFW    = os.path.join(BASE, "external", "faces_other_raw", "lfw-deepfunneled")
DATASET    = os.path.join(BASE, "dataset")

# PARÁMETROS
IMG_SIZE = (224,224)
SEED = 42
TARGET_ANDRES_MAX = 1500   # mis imágenes
CALTECH_TARGET    = 1000   # fondos
LFW_TARGET        = 500    # caras ajenas

# categorías Caltech sin personas; incluye BACKGROUND_Google
CALTECH_BG_CATEGORIES = [
    "BACKGROUND_Google","accordion","airplanes","anchor","ant","barrel","Buddha",
    "bonsai","camera","car_side","chair","chandelier","cup","dolphin","elephant",
    "ewer","ferry","flower","garfield","gramophone","hawksbill","headphone",
    "helicopter","joshua_tree","kangaroo","ketch","keyboard","laptop","Leopards",
    "lotus","menorah","metronome","motorbike","nautilus","pyramid","revolver",
    "rhino","rooster","saxophone","schooner","scissors","soccer_ball","stapler",
    "starfish","stop_sign","sunflower","tick","trilobite","umbrella","watch",
    "water_lilly","yin_yang"
]  # evitamos Faces / Faces_easy

def ensure_dirs():
    for p in [
        os.path.join(DATASET,"train","Andres"),
        os.path.join(DATASET,"train","Fondo"),
        os.path.join(DATASET,"val","Andres"),
        os.path.join(DATASET,"val","Fondo"),
        os.path.join(DATASET,"test","Andres"),
        os.path.join(DATASET,"test","Fondo"),
    ]:
        os.makedirs(p, exist_ok=True)

def list_images(folder):
    if not os.path.isdir(folder): return []
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    return [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def write_224(dst, bgr):
    if bgr is None: return False
    if bgr.shape[1] != IMG_SIZE[0] or bgr.shape[0] != IMG_SIZE[1]:
        bgr = cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.imwrite(dst, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def clean_dataset():
    for sub in ["train","val","test"]:
        for cls in ["Andres","Fondo"]:
            d = os.path.join(DATASET, sub, cls)
            if os.path.isdir(d):
                for f in list_images(d):
                    try: os.remove(f)
                    except: pass

def split_70_15_15(lst):
    n = len(lst)
    n_train = math.floor(0.70*n)
    n_val   = math.floor(0.15*n)
    n_test  = n - n_train - n_val
    return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

def copy_list(src_files, dst_dir, resize_if_needed=False, desc="Copiando"):
    os.makedirs(dst_dir, exist_ok=True)
    for src in tqdm(src_files, desc=desc):
        dst = os.path.join(dst_dir, os.path.basename(src))
        if resize_if_needed:
            img = cv2.imread(src)
            write_224(dst, img)
        else:
            shutil.copy2(src, dst)

# Selección VARIADA por video ===
VIDEO_RE = re.compile(r"^andres_(.+?)_\d+\.(jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)

def interleave_by_video(files, limit):
    buckets = {}
    for f in files:
        name = os.path.basename(f)
        m = VIDEO_RE.match(name)
        key = m.group(1) if m else "unknown"
        buckets.setdefault(key, []).append(f)
    for k in buckets:
        buckets[k].sort()  # mantiene orden temporal por video

    keys = list(buckets.keys())
    random.shuffle(keys)

    max_per_video = math.ceil(limit / max(1, len(keys))) + 2

    selected = []
    while len(selected) < limit:
        progressed = False
        for k in keys:
            lst = buckets[k]
            if not lst: 
                continue
            taken_from_k = sum(1 for s in selected if VIDEO_RE.match(os.path.basename(s)) and VIDEO_RE.match(os.path.basename(s)).group(1)==k)
            if taken_from_k >= max_per_video:
                continue
            selected.append(lst.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected

# Fondo: Caltech (escenas) + LFW (caras ajenas)
def sample_caltech(n):
    if not os.path.isdir(RAW_BG):
        raise RuntimeError(f"No encuentro Caltech-101 en {RAW_BG}")
    pool = []
    for c in CALTECH_BG_CATEGORIES:
        cat_dir = os.path.join(RAW_BG, c)
        if os.path.isdir(cat_dir):
            pool += glob(os.path.join(cat_dir, "*.jpg"))
    if len(pool) < n:
        raise RuntimeError(f"Caltech insuficiente: necesito {n}, hay {len(pool)}")
    random.shuffle(pool)
    return pool[:n]

def sample_lfw(n):
    if not os.path.isdir(RAW_LFW):
        raise RuntimeError(f"No encuentro LFW en {RAW_LFW}")
    people = [d for d in glob(os.path.join(RAW_LFW, "*")) if os.path.isdir(d)]
    pool = []
    for p in people:
        pool += glob(os.path.join(p, "*.jpg"))
    if len(pool) < n:
        raise RuntimeError(f"LFW insuficiente: necesito {n}, hay {len(pool)}")
    random.shuffle(pool)
    return pool[:n]

def main():
    random.seed(SEED); np.random.seed(SEED)
    ensure_dirs()

    # A) Seleccionar 1500 intercalando por video
    andres_all = list_images(SRC_ANDRES)
    if not andres_all:
        raise RuntimeError(f"No hay imágenes en {SRC_ANDRES}")
    random.shuffle(andres_all)
    andres_sel = interleave_by_video(andres_all, min(TARGET_ANDRES_MAX, len(andres_all)))
    need_fondo = len(andres_sel)

    # B) FONDO: 1000 Caltech + 500 LFW
    caltech_need = min(CALTECH_TARGET, need_fondo)
    lfw_need     = min(LFW_TARGET, need_fondo - caltech_need)
    if caltech_need + lfw_need < need_fondo:
        caltech_need = need_fondo - lfw_need

    caltech_sel = sample_caltech(caltech_need)
    lfw_sel     = sample_lfw(lfw_need)
    fondo_sel   = caltech_sel + lfw_sel
    random.shuffle(fondo_sel)

    # C) Split 70/15/15
    A_tr, A_val, A_te = split_70_15_15(andres_sel)
    F_tr, F_val, F_te = split_70_15_15(fondo_sel)

    # D) Copiar (Andres ya 224; Fondo redimensiona)
    clean_dataset()
    copy_list(A_tr, os.path.join(DATASET,"train","Andres"), resize_if_needed=False, desc="train/Andres")
    copy_list(F_tr, os.path.join(DATASET,"train","Fondo"),  resize_if_needed=True,  desc="train/Fondo")
    copy_list(A_val, os.path.join(DATASET,"val","Andres"),  resize_if_needed=False, desc="val/Andres")
    copy_list(F_val, os.path.join(DATASET,"val","Fondo"),   resize_if_needed=True,  desc="val/Fondo")
    copy_list(A_te, os.path.join(DATASET,"test","Andres"),  resize_if_needed=False, desc="test/Andres")
    copy_list(F_te, os.path.join(DATASET,"test","Fondo"),   resize_if_needed=True,  desc="test/Fondo")

    # Resumen
    def cnt(p): return len(list_images(p))
    print("\n[RESUMEN DEL DATASET]")
    for part in ["train","val","test"]:
        a = cnt(os.path.join(DATASET, part, "Andres"))
        f = cnt(os.path.join(DATASET, part, "Fondo"))
        print(f"{part}: Andres={a} | Fondo={f}")
    print(f"\nDetalles -> Andres_sel={len(andres_sel)}, Fondo_sel={len(fondo_sel)} (Caltech={len(caltech_sel)}, LFW={len(lfw_sel)})")

if __name__ == "__main__":
    main()
