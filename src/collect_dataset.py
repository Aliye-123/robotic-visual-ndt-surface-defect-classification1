import cv2
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "dataset", "dataset_endustriyel_kare")

CLASSES = {
    "0": "unknown",
    "1": "boyali_hasarsiz",
    "2": "boyali_orta_hasarli",
    "3": "boyali_yuksek_hasarli",
    "4": "boyasiz_hasarsiz",
    "5": "boyasiz_orta_hasarli",
    "6": "boyasiz_yuksek_hasarli",
}

WINDOW_NAME = "Kare Veri Toplama"

OUTER_BOX_SIZE = 340

INNER_BOX_SIZE = 240
os.makedirs(BASE_DIR, exist_ok=True)

for class_name in CLASSES.values():
    os.makedirs(os.path.join(BASE_DIR, class_name), exist_ok=True)

image_counts = {}
for class_name in CLASSES.values():
    folder = os.path.join(BASE_DIR, class_name)
    image_counts[class_name] = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acilamadi.")
    raise SystemExit

active_class = None

print("=== KARE VERI TOPLAMA ===")
print("0 -> unknown")
print("1 -> boyali_hasarsiz")
print("2 -> boyali_orta_hasarli")
print("3 -> boyali_yuksek_hasarli")
print("4 -> boyasiz_hasarsiz")
print("5 -> boyasiz_orta_hasarli")
print("6 -> boyasiz_yuksek_hasarli")
print("7 -> dokum_hasarsiz")
print("8 -> dokum_hasarli")
print("9 -> kaynak_dikisi_var")
print("a -> kaynak_dikisi_yok")
print("s -> sadece merkez kareyi kaydet")
print("q -> cikis")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Goruntu alinamadi.")
        break

    display_frame = frame.copy()
    h, w = frame.shape[:2]

    cx, cy = w // 2, h // 2

    # Dis kare
    ox1 = cx - OUTER_BOX_SIZE // 2
    oy1 = cy - OUTER_BOX_SIZE // 2
    ox2 = ox1 + OUTER_BOX_SIZE
    oy2 = oy1 + OUTER_BOX_SIZE

    # Ic kare (kaydedilecek alan)
    ix1 = cx - INNER_BOX_SIZE // 2
    iy1 = cy - INNER_BOX_SIZE // 2
    ix2 = ix1 + INNER_BOX_SIZE
    iy2 = iy1 + INNER_BOX_SIZE

    cv2.rectangle(display_frame, (ox1, oy1), (ox2, oy2), (0, 255, 255), 2)   # sari
    cv2.rectangle(display_frame, (ix1, iy1), (ix2, iy2), (255, 0, 255), 2)   # mor

    roi = frame[iy1:iy2, ix1:ix2]

    # Yazilar
    cv2.putText(display_frame, "Parcayi sari karenin icine yerlestir", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.putText(display_frame, "Kayit edilen alan: mor kare", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    cv2.putText(display_frame, "Elini mor kare disinda tutmaya calis", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    if active_class is not None:
        cv2.putText(display_frame, f"Aktif sinif: {active_class}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Adet: {image_counts[active_class]}", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(display_frame, "Once sinif sec (0-9 veya a)", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    guide = [
        "0: unknown",
        "1: boyali_hasarsiz",
        "2: boyali_orta_hasarli",
        "3: boyali_yuksek_hasarli",
        "4: boyasiz_hasarsiz",
        "5: boyasiz_orta_hasarli",
        "6: boyasiz_yuksek_hasarli",
        "7: dokum_hasarsiz",
        "8: dokum_hasarli",
        "9: kaynak_dikisi_var",
        "a: kaynak_dikisi_yok",
        "s: mor kareyi kaydet",
        "q: cikis"
    ]

    start_y = 200
    for i, line in enumerate(guide):
        cv2.putText(display_frame, line, (20, start_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF
    key_char = chr(key).lower() if 32 <= key <= 126 else ""

    if key_char in CLASSES:
        active_class = CLASSES[key_char]
        print(f"Aktif sinif: {active_class}")

    elif key_char == "s":
        if active_class is None:
            print("Once sinif sec!")
            continue

        image_counts[active_class] += 1
        filename = f"{active_class}_{image_counts[active_class]:04d}.jpg"
        save_path = os.path.join(BASE_DIR, active_class, filename)

        cv2.imwrite(save_path, roi)
        print(f"Kaydedildi: {save_path}")

    elif key_char == "q":
        print("Program kapatildi.")
        break

cap.release()
cv2.destroyAllWindows()
