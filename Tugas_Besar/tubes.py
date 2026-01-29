import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load Model
MODEL_PATH = r"C:\Users\INDRA WIJAYA\Documents\smt 3\Visi Komputer\runs_lama\detect_lama\train_lama\weights\best_lama.pt"
model = YOLO(MODEL_PATH)

# Variabel untuk menstabilkan grid (Smoothing)
last_coords = None  # Menyimpan koordinat (x1, y1, x2, y2) terakhir yang valid
alpha = 0.2         # Faktor smoothing (0.1 - 0.5), semakin kecil semakin stabil/lambat

COLOR_RANGES_BACKUP = {
    "red":    [((0, 160, 60), (7, 255, 255)), ((160, 160, 60), (180, 255, 255))],
    "orange": [((8, 160, 60), (22, 255, 255))],
    "yellow": [((23, 100, 100), (35, 255, 255))],
    "green":  [((36, 100, 60), (85, 255, 255))],
    "blue":   [((90, 160, 60), (128, 255, 255))],
    "white":  [((0, 0, 160), (180, 50, 255))],
}

COLOR_MAP = {
    "blue": (255, 0, 0), "green": (0, 255, 0), "orange": (0, 165, 255),
    "red": (0, 0, 255), "white": (255, 255, 255), "yellow": (0, 255, 255),
    "unknown": (50, 50, 50)
}

def get_backup_color(roi):
    if roi.size == 0: return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hsv = np.median(hsv.reshape(-1, 3), axis=0)
    
    lower_w, upper_w = COLOR_RANGES_BACKUP["white"][0]
    if np.all(avg_hsv >= lower_w) and np.all(avg_hsv <= upper_w):
        return "white"

    for color, ranges in COLOR_RANGES_BACKUP.items():
        if color == "white": continue
        for (lower, upper) in ranges:
            if np.all(avg_hsv >= lower) and np.all(avg_hsv <= upper):
                return color
    return "unknown"

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h_f, w_f, _ = frame.shape
    side_gui = np.zeros((h_f, 350, 3), dtype=np.uint8)
    
    # Deteksi YOLO dengan confidence lebih rendah agar tetap berusaha mencari saat warna solid
    results = model(frame, conf=0.25, iou=0.45, verbose=False)

    detected_stickers = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])].lower()
            detected_stickers.append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'label': label
            })

    # --- LOGIKA PENSTABIL GRID ---
    current_box = None
    if len(detected_stickers) >= 2: # Minimal 2 stiker untuk membentuk area
        coords = np.array([s['bbox'] for s in detected_stickers])
        x_min_curr, y_min_curr = np.min(coords[:, 0]), np.min(coords[:, 1])
        x_max_curr, y_max_curr = np.max(coords[:, 2]), np.max(coords[:, 3])
        current_box = np.array([x_min_curr, y_min_curr, x_max_curr, y_max_curr])

        if last_coords is None:
            last_coords = current_box
        else:
            # Low-pass filter agar grid tidak "melompat-lompat"
            last_coords = (alpha * current_box + (1 - alpha) * last_coords).astype(int)

    # Gunakan koordinat terakhir jika deteksi sekarang gagal/sedikit
    if last_coords is not None:
        x_min, y_min, x_max, y_max = last_coords
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        cell_w = (x_max - x_min) // 3
        cell_h = (y_max - y_min) // 3

        for i in range(3):
            for j in range(3):
                cx1, cy1 = x_min + (j * cell_w), y_min + (i * cell_h)
                cx2, cy2 = cx1 + cell_w, cy1 + cell_h
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 1)

                # Prioritas YOLO
                color_label = "unknown"
                for s in detected_stickers:
                    scx, scy = s['center']
                    if cx1 <= scx <= cx2 and cy1 <= scy <= cy2:
                        color_label = s['label']
                        break

                # Backup HSV jika YOLO gagal deteksi di area grid ini
                if color_label == "unknown":
                    # Sample lebih kecil di tengah agar tidak goyang
                    roi_backup = frame[cy1+15:cy2-15, cx1+15:cx2-15]
                    color_label = get_backup_color(roi_backup)

                # Update GUI
                box_size = 70
                start_x, start_y = 70 + (j * box_size), 100 + (i * box_size)
                current_color = COLOR_MAP.get(color_label, COLOR_MAP["unknown"])
                cv2.rectangle(side_gui, (start_x, start_y), (start_x + box_size - 5, start_y + box_size - 5), current_color, -1)
                cv2.rectangle(side_gui, (start_x, start_y), (start_x + box_size - 5, start_y + box_size - 5), (255, 255, 255), 1)

    final_output = np.hstack((frame, side_gui))
    cv2.imshow("Rubik Hybrid - Stable Grid", final_output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()