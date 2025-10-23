# Detecta posibles imágenes casi idénticas entre splits (solo clase Andres)
from pathlib import Path
from PIL import Image
import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "dataset"
CLASS = "Andres"

HASH_SIZE = 8
HAMMING_THR = 4  # más estricto: <=4 se considera "muy similar"

def dhash(path, hash_size=HASH_SIZE):
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((hash_size+1, hash_size), Image.Resampling.LANCZOS)
            diff = np.array(img)[:,1:] > np.array(img)[:,:-1]
            bit_string = 0
            for row in diff:
                for v in row: bit_string = (bit_string << 1) | int(v)
            return bit_string
    except Exception:
        return None

def ham(a,b): return bin(a ^ b).count("1")

def collect(split):
    root = DATA / split / CLASS
    files = [p for p in root.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")]
    hashes=[]
    for p in files:
        h=dhash(p)
        if h is not None: hashes.append((p,h))
    return hashes

def main():
    sets = {s: collect(s) for s in ("train","val","test")}
    pairs = [("train","val"),("train","test"),("val","test")]
    leaks=[]
    for a,b in pairs:
        for pa,ha in sets[a]:
            for pb,hb in sets[b]:
                if ham(ha,hb) <= HAMMING_THR:
                    leaks.append((a,b,pa.name,pb.name))
    if not leaks:
        print("[OK] Sin near-duplicates entre splits en clase Andres.")
    else:
        print(f"[WARN] Posibles fugas ({len(leaks)}):")
        for a,b,fa,fb in leaks[:50]:
            print(f"{a} ↔ {b} | {fa}  ~  {fb}")
        if len(leaks)>50: print("...")

if __name__ == "__main__":
    main()
