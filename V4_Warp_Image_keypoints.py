import cv2
from V4_SimulateBoardOnWarpedImageKey import run_simulation
from V4_Extract_DartCenters import extract_dart_keypoints
from V4_Extract_DartCenters import transform_dart_keypoints_absolute
import os, json, cv2, numpy as np
from V4_YOLODartKoordinates import run_yolo_on_image
from V4_YOLODartKoordinates import run_yolo_on_image2
from V4_DartJsonLogic import update_dart_json
# -----------------------------
# Parameter
# -----------------------------

def Process_Start_Main(img, out_dir="out", fallback_score=10):

    """
    Nimmt ein OpenCV-Image, verarbeitet es und gibt den Score zurück.
    Falls keine gültige Erkennung möglich ist, wird fallback_score zurückgegeben.
    """
    try:
        print("hi")
        #IMG_W, IMG_H = img.shape[1], img.shape[0]
        SIZE = 400
        FLIP_X = False
        ANGLE_CORRECTION = 9
        json_path= r"UsefulScripts\WichtigeScripts\Aktuell\Versuch4\keys.json"
        os.makedirs(out_dir, exist_ok=True)
        OUT_PATH_FILE = os.path.join(out_dir, "result_img.jpg")

        # --- Board Keypoints via YOLO ---
        if is_json_empty(json_path):

            keypoints = run_yolo_on_image(r"runs\AktuellBest\Board.pt", img, wert=False)
            if len(keypoints) < 4:
                print("⚠️ Nicht genug Board-Keypoints gefunden – Fallback")
                return 
            data= {
                "top": keypoints[0],
                "right": keypoints[1],
                "bottom": keypoints[2],
                "left": keypoints[3]
            }
            with open(json_path, "w") as f:
                json.dump(data, f,indent=2)
            TOP, RIGHT, BOTTOM, LEFT = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
        
        else:
            print("🟢 Verwende gespeicherte Keypoints aus JSON.")
            with open(json_path, "r") as f:
                data = json.load(f)

            TOP = data["top"]
            RIGHT = data["right"]
            BOTTOM = data["bottom"]
            LEFT = data["left"]

        # --- Zielkoordinaten ---
        dst_pts = np.array([
            [SIZE/2, 0],
            [SIZE, SIZE/2],
            [SIZE/2, SIZE],
            [0, SIZE/2]
        ], dtype=np.float32)

        # --- Rotation ---
        M = cv2.getRotationMatrix2D((SIZE/2, SIZE/2), ANGLE_CORRECTION, 1.0)
        ones = np.ones((dst_pts.shape[0], 1))
        dst_hom = np.hstack([dst_pts, ones])
        dst_rot = (M @ dst_hom.T).T.astype(np.float32)

        src_pts = np.array([TOP, RIGHT, BOTTOM, LEFT], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_rot)

        # --- Bild entzerren ---
        warped = cv2.warpPerspective(img, H, (SIZE, SIZE))
        cv2.imwrite(OUT_PATH_FILE, warped)

        # --- Darts erkennen ---
        dart_hits_raw = run_yolo_on_image2(r"C:\Users\Josi\Documents\DartProgramm\runs\pose\train3\weights\best.pt", img, wert=False)
        if not dart_hits_raw:
            print("⚠️ Keine Darts gefunden – Fallback")
            return 

        dart_hits = transform_dart_keypoints_absolute(
            dart_hits_raw, H, M, SIZE=SIZE, FLIP_X=FLIP_X
        )


        # --- Darts speichern & prüfen ---
        saved_darts, ready = update_dart_json(dart_hits)
        if not ready:
            return None  # Noch nicht genug Darts

        # --- Testpunkte & Score ---
        dart_hits = [(d["x"], d["y"]) for d in saved_darts]
        test_points = {
            f"Dart {i}": (int(x), int(y)) for i, (x, y) in enumerate(dart_hits, start=1)
        }

        # --- Simulation & Score ---
        resultscore = run_simulation(warped, out_dir, test_points=test_points)

        if resultscore is None:
            print("⚠️ Score konnte nicht berechnet werden – Fallback")
            return 

        return resultscore

    except Exception as e:
        print(f"⚠️ Fehler: {e} – Fallback")
        return  fallback_score



import os, json

def is_json_empty(file_path: str) -> bool:
    """
    Prüft, ob eine JSON-Datei leer, ungültig oder unvollständig ist.
    Gibt True zurück, wenn:
      - sie nicht existiert,
      - sie 0 Bytes groß ist,
      - ihr Inhalt leer ({} / []) ist,
      - alle Werte None/null sind,
      - oder weniger als 4 gültige Keypoints vorhanden sind.
    """
    if not os.path.exists(file_path):
        return True

    if os.stat(file_path).st_size == 0:
        return True

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

            # 🔸 Kein Inhalt oder leeres Dict
            if not data:
                return True

            # 🔸 Erwartete Keys prüfen
            required_keys = {"top", "right", "bottom", "left"}
            if not all(k in data for k in required_keys):
                print("⚠️ JSON unvollständig – fehlende Keys erkannt.")
                return True

            # 🔸 Prüfen, ob alle Werte gültig sind (nicht None, nicht leer)
            valid_values = [
                v for v in data.values()
                if v is not None and isinstance(v, (list, tuple)) and len(v) == 2
            ]

            if len(valid_values) != 4:
                print("⚠️ JSON enthält nicht 4 gültige Keypoints.")
                return True

    except json.JSONDecodeError:
        print("⚠️ JSON konnte nicht gelesen werden – Formatfehler.")
        return True

    # ✅ Alle Bedingungen erfüllt → Datei ist gültig
    return False


def clear_json(json_path):
    """
    Setzt die JSON-Datei auf leeren Inhalt ({}).
    Falls sie nicht existiert, wird sie neu angelegt.
    """
    with open(json_path, "w") as f:
        json.dump({}, f)
    print(f"🧹 JSON-Inhalt geleert: {json_path}")

if __name__ == "__main__":
    clear_json(json_path= r"UsefulScripts\WichtigeScripts\Aktuell\Versuch4\keys.json")