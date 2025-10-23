# download_lookalikes.py
import os
from icrawler.builtin import BingImageCrawler

BASE = r"D:\Archivos de Usuario\Documents\Artificial-Intelligence\Lab05"
RAW_DIR = os.path.join(BASE, "external", "lookalikes_raw")

# Personas y consultas de búsqueda
PEOPLE = {
    "charlie_zaa":      ['"Charlie Zaa" rostro', '"Charlie Zaa" close up', '"Charlie Zaa" portrait'],
    "kalimba":          ['"Kalimba Marichal" rostro', '"Kalimba" portrait'],
    "tenoch_huerta":    ['"Tenoch Huerta" rostro', '"Tenoch Huerta" close up'],
    "jorge_celedon":    ['"Jorge Celedón" rostro', '"Jorge Celedon" portrait'],
    "fonseca":          ['"Fonseca" cantante rostro', '"Fonseca" portrait'],
    "andres_cepeda":    ['"Andrés Cepeda" rostro', '"Andres Cepeda" portrait'],
}

IMAGES_PER_QUERY = 60  # intenta bajar ~60 por consulta; luego recortamos y filtramos

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def download_person(person_key, queries):
    out_dir = os.path.join(RAW_DIR, person_key)
    ensure_dir(out_dir)
    for q in queries:
        crawler = BingImageCrawler(storage={"root_dir": out_dir})
        crawler.crawl(keyword=q, max_num=IMAGES_PER_QUERY, file_idx_offset="auto")

def main():
    ensure_dir(RAW_DIR)
    for person, queries in PEOPLE.items():
        print(f"[DESCARGANDO] {person} ...")
        download_person(person, queries)
    print("\n[OK] Descarga terminada. Revisa external\\lookalikes_raw\\*")

if __name__ == "__main__":
    main()
