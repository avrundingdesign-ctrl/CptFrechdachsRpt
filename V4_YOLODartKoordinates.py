from ultralytics import YOLO
import numpy as np

def load_model(model_path):
    """Lädt ein YOLO-Modell einmalig und gibt es zurück."""
    print(f"📦 Lade YOLO-Modell: {model_path}")
    return YOLO(model_path)

# mit wert kann man umschalten zwischen dem DartTraining und dem BoardTraining, bzw bisher vorher festgelegt da best.pt schlechte ergebnisse liefert
def run_yolo_on_image(model, image_path, wert, out_txt="Boardresults.txt", imgsz=800):
    """
    Führt YOLO-Inferenz auf einem Bild aus, gibt ALLE Keypoints als flache Liste zurück
    plus die zugehörigen Detection-Confidences und speichert die Koordinaten
    zusätzlich in einer Textdatei.
    model = bereits geladenes YOLO-Modell (kein Pfad mehr!)
    """
    results = model.predict(source=image_path, imgsz=imgsz, verbose=False, save=False, save_txt=False)

    # Hier speichern wir erstmal (cls, (x,y), confidence).
    # YOLO liefert hier bereits nur Detections >= internem Confidence-Threshold zurück.
    keypoints_with_cls = []

    for r in results:
        if r.keypoints is not None and r.boxes is not None:
            for box, kp_xy in zip(r.boxes, r.keypoints.xy):
                cls_id = int(box.cls[0].item())
                x, y = kp_xy[0].tolist()
                confidence = float(box.conf[0].item()) if box.conf is not None else None
                keypoints_with_cls.append((cls_id, (float(x), float(y)), confidence))


    # sortieren nach Klassen-IDs (0–3)
    keypoints_all, confidences_all = sort_by_class(keypoints_with_cls)

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
        confidences_all = [1.0, 1.0, 1.0, 1.0]
        print(keypoints_all)

    return keypoints_all, confidences_all


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
    Erwartet eine Liste [(cls_id, (x,y), confidence), ...] und sortiert nach Klasse 0–3.
    Gibt zwei Listen zurück:
      - [kp0, kp1, kp2, kp3]
      - [conf0, conf1, conf2, conf3]
    fehlende Klassen = None.
    """
    ordered = [None, None, None, None]
    ordered_confidences = [None, None, None, None]
    for cls_id, (x, y), confidence in points_with_cls:
        if 0 <= cls_id < 4:
            ordered[cls_id] = (x, y)
            ordered_confidences[cls_id] = confidence
    return ordered, ordered_confidences

def run_yolo_on_image2(model, image_path,wert, out_txt="Boardresults.txt", imgsz=800):
    
    """
    Führt YOLO-Inferenz auf einem Bild aus, gibt ALLE Keypoints als flache Liste zurück
    plus die zugehörigen Detection-Confidences und speichert die Koordinaten
    zusätzlich in einer Textdatei.
    model = bereits geladenes YOLO-Modell (kein Pfad mehr!)
    """
    results = model.predict(source=image_path, imgsz=imgsz, verbose=False,save=False, save_txt=False)

    keypoints_with_confidence = []

    for r in results:
        if r.keypoints is not None and r.boxes is not None:
            for box, kp_xy in zip(r.boxes, r.keypoints.xy):   # kp_xy = Tensor (K x 2)
                confidence = float(box.conf[0].item()) if box.conf is not None else None
                for (x, y) in kp_xy.tolist():
                    keypoints_with_confidence.append((float(x), float(y), confidence))

    # --- Ergebnisse in Datei schreiben ---
    with open(out_txt, "w") as f:
        for (x, y, _) in keypoints_with_confidence:
            f.write(f"{x:.3f},{y:.3f}\n")

    print(keypoints_with_confidence)
    if wert:
        #keypoints_all = sort_TRBL (keypoints_all)
        keypoints_with_confidence = [
            (1188.15380859375, 1192.018798828125, 1.0),
            (1522.02783203125, 2035.2091064453125, 1.0),
            (1762.0284423828125, 2951.6484375, 1.0),
            (549.4423217773438, 2785.3603515625, 1.0)
        ]
        print(keypoints_with_confidence)
    
    return keypoints_with_confidence