from ultralytics import YOLO
import numpy as np

# mit wert kann man umschalten zwischen dem DartTraining und dem BoardTraining, bzw bisher vorher festgelegt da best.pt schlechte ergebnisse liefert
def run_yolo_on_image(model_path, image_path, wert, out_txt="Boardresults.txt", imgsz=800):
    """
    Führt YOLO-Inferenz auf einem Bild aus, gibt ALLE Keypoints als flache Liste zurück
    (ohne Confidence) und speichert sie zusätzlich in einer Textdatei.
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=imgsz, verbose=False, save=True, save_txt=True)

    # hier speichern wir erstmal (cls, (x,y))
    keypoints_with_cls = []

    for r in results:
        if r.keypoints is not None and r.boxes is not None:
            for box, kp_xy in zip(r.boxes, r.keypoints.xy):
                cls_id = int(box.cls[0].item())
                x, y = kp_xy[0].tolist()
                keypoints_with_cls.append((cls_id, (float(x), float(y))))


    # sortieren nach Klassen-IDs (0–3)
    keypoints_all = sort_by_class(keypoints_with_cls)

    # --- Ergebnisse in Datei schreiben ---
    with open(out_txt, "w") as f:
        for i, kp in enumerate(keypoints_all):
            if kp is not None:
                f.write(f"{i}: {kp[0]:.3f},{kp[1]:.3f}\n")

    print("Sortierte Keypoints:", keypoints_all)

    if wert:
        # Testwerte wie gehabt
        keypoints_all = [
            (1188.15380859375, 1192.018798828125),
            (1522.02783203125, 2035.2091064453125),
            (1762.0284423828125, 2951.6484375),
            (549.4423217773438, 2785.3603515625)
        ]
        print(keypoints_all)

    return keypoints_all


def sort_TRBL(points):
    pts = np.array(points)
    cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])

    order = [None, None, None, None]  # [TOP, RIGHT, BOTTOM, LEFT]

    for (x, y) in pts:
        dx, dy = x - cx, y - cy
        angle = np.degrees(np.arctan2(dy, dx))

        if -45 <= angle < 45:
            order[1] = (float(x), float(y))   # RIGHT
        elif 45 <= angle < 135:
            order[2] = (float(x), float(y))   # BOTTOM
        elif angle >= 135 or angle < -135:
            order[3] = (float(x), float(y))   # LEFT
        else:
            order[0] = (float(x), float(y))   # TOP

    return order


def sort_by_class(points_with_cls):
    """
    Erwartet eine Liste [(cls_id, (x,y)), ...] und sortiert nach Klasse 0–3.
    Gibt eine Liste [kp0, kp1, kp2, kp3] zurück, fehlende Klassen = None.
    """
    ordered = [None, None, None, None]
    for cls_id, (x, y) in points_with_cls:
        if 0 <= cls_id < 4:
            ordered[cls_id] = (x, y)
    return ordered

def run_yolo_on_image2(model_path, image_path,wert, out_txt="Boardresults.txt", imgsz=800):
    
    """
    FÃ¼hrt YOLO-Inferenz auf einem Bild aus, gibt ALLE Keypoints als flache Liste zurÃ¼ck
    (ohne Confidence) und speichert sie zusÃ¤tzlich in einer Textdatei.
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=imgsz, verbose=False,save=True, save_txt=True)

    keypoints_all = []

    for r in results:
        if r.keypoints is not None:
            for kp_xy in r.keypoints.xy:   # kp_xy = Tensor (K x 2)
                for (x, y) in kp_xy.tolist():
                    keypoints_all.append((float(x), float(y)))

    # --- Ergebnisse in Datei schreiben ---
    with open(out_txt, "w") as f:
        for (x, y) in keypoints_all:
            f.write(f"{x:.3f},{y:.3f}\n")

    print(keypoints_all)
    if wert:
        #keypoints_all = sort_TRBL (keypoints_all)
        keypoints_all = [(1188.15380859375, 1192.018798828125), (1522.02783203125, 2035.2091064453125), (1762.0284423828125, 2951.6484375), (549.4423217773438, 2785.3603515625)]
        print(keypoints_all)
    
    return keypoints_all

    return keypoints_all