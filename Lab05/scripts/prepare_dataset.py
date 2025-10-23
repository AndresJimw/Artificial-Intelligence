# Prepara dataset binario: Andres vs Fondo
# - Agrupa tus imágenes por minuto de captura (evita fuga de frames parecidos)
# - Usa ~95% de lookalikes como negativos (Fondo)
# - Balancea 1:1 por split (train, val, test)

import os, re, cv2, random
from pathlib import Path

# ---------- Configuración ----------
SEED = 42
IMG_SIZE = (224, 224)
SPLIT = (0.70, 0.15, 0.15)
BALANCE_RATIO = 1.0

R_FONDO = {"lookalikes": 0.95, "caltech": 0.05, "lfw": 0.00}
CAP_LOOKALIKE_PER_PERSON = 150
CAP_LFW_PER_PERSON = 20

# ---------- Rutas ----------
BASE = Path(__file__).resolve().parents[1]
P_AND = BASE / "andres_crops_224"
P_LK  = BASE / "lookalikes_crops_224"
P_LFW = BASE / "external" / "faces_other_raw" / "lfw-deepfunneled"
P_CAL = BASE / "external" / "backgrounds_raw" / "101_ObjectCategories"
DATA  = BASE / "dataset"
EXTS = (".jpg",".jpeg",".png",".bmp",".webp")

# ---------- Funciones auxiliares ----------
def list_images_recursive(folder: Path):
    if not folder.is_dir(): return []
    out = []
    for r, _, fs in os.walk(folder):
        rp = Path(r)
        if any(seg.lower() == "duplicates" for seg in rp.parts):
            continue
        for f in fs:
            if f.lower().endswith(EXTS):
                out.append(rp / f)
    return out

def write_224(dst: Path, bgr):
    if bgr is None: return False
    if (bgr.shape[1], bgr.shape[0]) != IMG_SIZE:
        bgr = cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def clean_dataset():
    if not DATA.exists(): return
    for part in ("train","val","test"):
        for cls in ("Andres","Fondo"):
            d = DATA / part / cls
            if d.is_dir():
                for p in d.glob("*"):
                    try: p.unlink()
                    except: pass

def ensure_dirs():
    for part in ("train","val","test"):
        for cls in ("Andres","Fondo"):
            (DATA/part/cls).mkdir(parents=True, exist_ok=True)

def split_counts(n, ratios):
    tr = int(round(n * ratios[0]))
    va = int(round(n * ratios[1]))
    te = max(0, n - tr - va)
    if tr == 0 and n > 0: tr = 1
    if va == 0 and n - tr > 1: va = 1
    te = max(0, n - tr - va)
    return tr, va, te

def assign_groups_to_splits(keys, ratios=SPLIT):
    keys = list(keys)
    random.shuffle(keys)
    n = len(keys)
    tr, va, te = split_counts(n, ratios)
    return {"train": set(keys[:tr]), "val": set(keys[tr:tr+va]), "test": set(keys[tr+va:tr+va+te])}

def flatten_selected(groups, sel_keys):
    out = []
    for k in sel_keys:
        out.extend(groups.get(k, []))
    random.shuffle(out)
    return out

def sample_caltech(n):
    if n <= 0 or not P_CAL.exists(): return []
    pool = []
    for cdir in P_CAL.iterdir():
        if cdir.is_dir():
            pool += [p for p in cdir.iterdir() if p.suffix.lower() in EXTS]
    random.shuffle(pool)
    return pool[:n]

# ---------- Agrupación de ANDRES por minuto ----------
# Ejemplo: andres_VID_20251022_171437_0014.jpg -> VID_20251022_1714
SESSION_RX = re.compile(r"VID[_\s]*(\d{8})[_\s]*(\d{4})", re.IGNORECASE)
def session_key(path: Path):
    name = path.stem
    m = SESSION_RX.search(name)
    if m:
        return f"VID_{m.group(1)}_{m.group(2)}"
    # fallback
    m2 = re.match(r"^(.*?)(_?\d{1,4})$", name)
    return (m2.group(1) if m2 else name[:24])

def group_andres_by_session(imgs):
    groups = {}
    for p in imgs:
        k = session_key(p)
        groups.setdefault(k, []).append(p)
    return groups

# ---------- Principal ----------
def main():
    random.seed(SEED)

    # 1) ANDRES agrupado por minuto
    A_all = list_images_recursive(P_AND)
    if not A_all:
        raise RuntimeError(f"No hay imágenes en {P_AND}")
    A_groups = group_andres_by_session(A_all)
    split_map = assign_groups_to_splits(A_groups.keys())
    A_tr = flatten_selected(A_groups, split_map["train"])
    A_va = flatten_selected(A_groups, split_map["val"])
    A_te = flatten_selected(A_groups, split_map["test"])
    nA = {"train": len(A_tr), "val": len(A_va), "test": len(A_te)}
    needF = {k: int(round(v * BALANCE_RATIO)) for k, v in nA.items()}

    # 2) LOOKALIKES como fondo principal (sin dividir por identidad)
    def list_lookalikes_pool(root: Path):
        pool = []
        if root.exists():
            for d in sorted([p for p in root.iterdir() if p.is_dir()]):
                imgs = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
                if not imgs: continue
                random.shuffle(imgs)
                pool.extend(imgs[:CAP_LOOKALIKE_PER_PERSON])
        random.shuffle(pool)
        return pool

    LK_POOL = list_lookalikes_pool(P_LK)

    def take_without_overlap(pool, k, taken):
        out = []
        for p in pool:
            if p in taken: continue
            out.append(p)
            if len(out) == k: break
        return out

    def compose_negatives(need_tr, need_va, need_te):
        ratio_lk  = R_FONDO.get("lookalikes", 0.95)
        ratio_lfw = R_FONDO.get("lfw", 0.00)
        taken = set()

        n_lk_tr = int(round(need_tr * ratio_lk))
        n_lk_va = int(round(need_va * ratio_lk))
        n_lk_te = int(round(need_te * ratio_lk))
        LK_tr = take_without_overlap(LK_POOL, n_lk_tr, taken); taken.update(LK_tr)
        LK_va = take_without_overlap(LK_POOL, n_lk_va, taken); taken.update(LK_va)
        LK_te = take_without_overlap(LK_POOL, n_lk_te, taken); taken.update(LK_te)

        def topup(current, target):
            need = max(0, target - len(current))
            if need == 0: return current
            extra = sample_caltech(need)
            return current + extra

        F_tr = topup(LK_tr, need_tr)
        F_va = topup(LK_va, need_va)
        F_te = topup(LK_te, need_te)
        random.shuffle(F_tr); random.shuffle(F_va); random.shuffle(F_te)
        return F_tr, F_va, F_te

    F_tr, F_va, F_te = compose_negatives(needF["train"], needF["val"], needF["test"])

    # 3) Escribir dataset
    clean_dataset()
    ensure_dirs()

    def dump(lst, dst):
        ok = 0
        for src in lst:
            img = cv2.imread(str(src))
            if img is None: continue
            if write_224(dst / src.name, img): ok += 1
        return ok

    k1 = dump(A_tr, DATA/"train"/"Andres"); k2 = dump(F_tr, DATA/"train"/"Fondo")
    k3 = dump(A_va, DATA/"val"/"Andres");   k4 = dump(F_va, DATA/"val"/"Fondo")
    k5 = dump(A_te, DATA/"test"/"Andres");  k6 = dump(F_te, DATA/"test"/"Fondo")

    print("\n[RESUMEN]")
    print(f"train | Andres={k1} Fondo={k2}")
    print(f"val   | Andres={k3} Fondo={k4}")
    print(f"test  | Andres={k5} Fondo={k6}")
    print(f"\nSplit por MINUTO (videos) → evita fuga de frames similares.")
    print(f"Fondo: lookalikes {R_FONDO['lookalikes']*100:.0f}% | Caltech {R_FONDO['caltech']*100:.0f}% | LFW {R_FONDO['lfw']*100:.0f}%")
    print(f"Dataset → {DATA}")

if __name__ == "__main__":
    main()
