import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

def compute_dhash(img_path, hash_size=8):
    """
    Devuelve un entero de 64 bits con el dHash de la imagen.
    dHash: escalado a (hash_size+1, hash_size), pasa a grises y compara columnas adyacentes.
    """
    try:
        with Image.open(img_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(img, dtype=np.int16)
            diff = pixels[:, 1:] > pixels[:, :-1]
            # Convierte la matriz booleana a entero bit a bit
            bit_string = 0
            for row in diff:
                for val in row:
                    bit_string = (bit_string << 1) | int(val)
            return bit_string
    except Exception:
        return None

def hamming_distance(h1, h2):
    return (h1 ^ h2).bit_count()

def variance_of_laplacian(img_path):
    """
    Métrica simple de nitidez: Varianza del Laplaciano (más alto = más nítida).
    """
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return -1
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except Exception:
        return -1

def is_image(p: Path, extensions):
    return p.suffix.lower() in extensions

def main():
    parser = argparse.ArgumentParser(description="Detecta y mueve imágenes duplicadas o muy parecidas usando dHash.")
    parser.add_argument("--root", type=str, required=True,
                        help=r"Carpeta con las imágenes (ej: D:\...\andres_crops_224)")
    parser.add_argument("--dest", type=str, default="duplicates",
                        help="Carpeta de destino para mover duplicados (se crea dentro de --root).")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Umbral de distancia Hamming (0=idénticas, 64=muy distintas). Recomendado 6–10.")
    parser.add_argument("--hash-size", type=int, default=8,
                        help="Tamaño del hash (8 => 64 bits). Dejar 8 normalmente.")
    parser.add_argument("--ext", type=str, default=".jpg,.jpeg,.png,.bmp,.webp",
                        help="Extensiones permitidas, separadas por coma.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo reporta, no mueve nada.")
    parser.add_argument("--keep", type=str, default="sharpest", choices=["sharpest", "first"],
                        help='Estrategia para conservar: "sharpest" (más nítida) o "first" (la primera encontrada).')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] No existe: {root}")
        return

    extensions = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    dest = root / args.dest
    dest.mkdir(parents=True, exist_ok=True)

    # 1) Indexamos imágenes y calculamos hashes
    files = [p for p in root.rglob("*") if p.is_file() and is_image(p, extensions) and dest not in p.parents]
    print(f"[INFO] Imágenes encontradas: {len(files)}")
    hashes = []
    valid_files = []

    for p in tqdm(files, desc="Calculando dHash"):
        h = compute_dhash(p, hash_size=args.hash_size)
        if h is not None:
            hashes.append(h)
            valid_files.append(p)
        else:
            print(f"[WARN] No se pudo leer: {p}")

    # 2) Agrupamos por similitud (distancia Hamming <= threshold) usando representantes
    #    Estrategia simple: iteramos y comparamos contra el representante de cada grupo.
    groups = []  # lista de listas de índices
    reps = []    # hash representante por grupo (el del primero conservado)

    for idx, h in enumerate(tqdm(hashes, desc="Agrupando similares")):
        placed = False
        for g_idx, rep_h in enumerate(reps):
            if hamming_distance(h, rep_h) <= args.threshold:
                groups[g_idx].append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])
            reps.append(h)

    # 3) Para cada grupo con más de 1, elegimos cuál conservar y cuáles mover
    duplicates_plan = []  # (src_path, dst_path)
    total_groups_multi = 0
    total_moves = 0

    for g in groups:
        if len(g) <= 1:
            continue
        total_groups_multi += 1

        # Elegimos el "keeper"
        if args.keep == "sharpest":
            sharpness = [(i, variance_of_laplacian(str(valid_files[i]))) for i in g]
            sharpness.sort(key=lambda x: x[1], reverse=True)
            keeper_idx = sharpness[0][0]
        else:  # "first"
            keeper_idx = g[0]

        group_paths = [valid_files[i] for i in g]
        keeper_path = valid_files[keeper_idx]

        for i in g:
            p = valid_files[i]
            if p == keeper_path:
                continue
            # destino con mismo nombre dentro de /duplicates
            rel = p.relative_to(root)
            dst = dest / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            duplicates_plan.append((p, dst))
            total_moves += 1

    # 4) Reporte
    print("\n========== RESUMEN ==========")
    print(f"Grupos totales: {len(groups)}")
    print(f"Grupos con duplicados: {total_groups_multi}")
    print(f"Imágenes a mover (duplicadas/casi duplicadas): {total_moves}")
    print(f"Destino: {dest}")
    print(f"Umbral Hamming: {args.threshold} (0=idénticas, 64=distintas)")
    print(f"Estrategia de conservación: {args.keep}")
    print(f"Modo simulación: {'SÍ' if args.dry_run else 'NO'}")
    print("=============================\n")

    # 5) Ejecutamos movimiento si no es dry-run
    if args.dry_run:
        print("[DRY-RUN] No se movió ningún archivo. Si estás conforme, ejecuta sin --dry-run.")
        return

    moved = 0
    for src, dst in tqdm(duplicates_plan, desc="Moviendo duplicados"):
        try:
            # Usar os.replace para permitir mover entre discos y sobrescribir si existe
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Para soportar rutas con caracteres no ASCII en Windows:
            data = np.fromfile(src, dtype=np.uint8)
            if data.size == 0:
                continue
            dst.write_bytes(data.tobytes())
            try:
                src.unlink()
            except Exception:
                pass
            moved += 1
        except Exception as e:
            print(f"[ERROR] No se pudo mover {src} -> {dst}: {e}")

    print(f"[OK] Movidas {moved} imágenes duplicadas a: {dest}")

if __name__ == "__main__":
    main()
