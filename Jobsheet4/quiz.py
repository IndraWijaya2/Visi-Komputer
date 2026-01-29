import cv2
import mediapipe as mp
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'hair_segmenter.tflite') 

def main():
    # 1. Cek apakah model ada sebelum lanjut
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR KERAS: File model tidak ditemukan di: {MODEL_PATH}")
        print("Pastikan file 'hair_segmenter.tflite' ada di folder yang sama dengan script ini.")
        return

    # 2. Setup Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Webcam tidak terdeteksi.")
        return

    # 3. Setup MediaPipe (Mode VIDEO)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO, 
        output_category_mask=True)

    # 4. Daftar Pilihan Warna (Format BGR)
    colors = [
        ("Original", None),
        ("Biru Elektrik", (255, 0, 0)),       
        ("Blonde Emas", (140, 215, 255)), 
        ("Merah Maroon", (0, 0, 180)),      
        ("Ungu", (255, 0, 255)),
        ("Hijau", (0, 200, 0))
    ]
    current_color_idx = 0

    print(f"\n=== PROGRAM BERJALAN ===")
    print(f"Model dimuat dari: {MODEL_PATH}")
    print("Tekan 'c' -> Ganti Warna")
    print("Tekan 'q' -> Keluar")

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontal (cermin)
            frame = cv2.flip(frame, 1)
            
            # Konversi frame ke MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Timestamp (Wajib untuk mode VIDEO)
            frame_timestamp_ms = int(time.time() * 1000)

            # --- PROSES SEGMENTASI ---
            segmentation_result = segmenter.segment_for_video(mp_image, frame_timestamp_ms)
            category_mask = segmentation_result.category_mask
            
            # Ambil data mask sebagai numpy array
            mask_np = category_mask.numpy_view()

            # --- LOGIKA KHUSUS (0=BG, 1=HAIR) ---
            # Kita hanya ambil yang nilainya 1
            hair_mask = np.where(mask_np == 1, 1, 0).astype(np.uint8)

            output_img = frame.copy()
            color_name, color_value = colors[current_color_idx]

            # Jika ada warna yang dipilih (bukan Original)
            if color_value is not None:
                # 1. Haluskan mask (penting agar tidak bergerigi)
                # Kalikan 255 agar jadi hitam-putih (0-255) untuk diblur
                hair_mask_blurred = cv2.GaussianBlur(hair_mask * 255, (7, 7), 0) / 255.0
                
                # 2. Buat mask 3 channel (H, W, 3)
                hair_mask_3d = np.repeat(hair_mask_blurred[:, :, np.newaxis], 3, axis=2)

                # 3. Buat kanvas warna solid
                colored_canvas = np.zeros_like(frame)
                colored_canvas[:] = color_value

                # 4. Extract area rambut dari frame asli & kanvas warna
                frame_hair = frame * hair_mask_3d
                color_hair = colored_canvas * hair_mask_3d
                
                # 5. Blending (Pencampuran)
                # Ubah angka 0.5 dan 0.5 untuk mengatur transparansi cat
                blended_hair = cv2.addWeighted(frame_hair.astype(np.uint8), 0.5, 
                                               color_hair.astype(np.uint8), 0.5, 0)
                
                # 6. Gabungkan Background + Rambut Baru
                background_mask = 1.0 - hair_mask_3d
                background = frame * background_mask
                
                output_img = (background + blended_hair).astype(np.uint8)

            # Tampilkan Informasi
            cv2.putText(output_img, f"Mode: {color_name}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow("Hair Segmentation Quiz", output_img)

            # Kontrol Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                current_color_idx = (current_color_idx + 1) % len(colors)
                print(f"Ganti warna ke: {colors[current_color_idx][0]}")
            elif key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()