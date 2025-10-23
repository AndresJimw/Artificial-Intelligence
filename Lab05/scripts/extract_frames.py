import os
import cv2
import argparse

def extract_frames_from_video(video_path, out_dir, fps_target=2):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] No pude abrir: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0
    frame_interval = int(round(src_fps / fps_target))
    if frame_interval <= 0:
        frame_interval = 1

    base = os.path.splitext(os.path.basename(video_path))[0]
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            out_name = f"{base}_{saved:04d}.jpg"
            out_path = os.path.join(out_dir, out_name)
            h, w = frame.shape[:2]
            max_side = max(h, w)
            if max_side > 2240:
                scale = 2240.0 / max_side
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_path, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"[OK] {os.path.basename(video_path)} -> {saved} fotos")
    return saved

def extract_from_dir(src_dir, out_dir, fps_target=2):
    total = 0
    for name in os.listdir(src_dir):
        if name.lower().endswith((".mp4",".mov",".mkv",".avi")):
            total += extract_frames_from_video(os.path.join(src_dir, name), out_dir, fps_target)
    print(f"\n[RESUMEN] Total de imágenes guardadas: {total}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Carpeta con videos")
    ap.add_argument("--dst", required=True, help="Carpeta de salida de imágenes")
    ap.add_argument("--fps", type=float, default=2.0, help="Fotos por segundo (default 2)")
    args = ap.parse_args()

    extract_from_dir(args.src, args.dst, args.fps)
