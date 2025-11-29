import os, cv2, numpy as np
from V4_SimulateBoardOnWarpedImageKey import run_simulation
from V4_Extract_DartCenters import transform_dart_keypoints_absolute
from V4_YOLODartKoordinates import run_yolo_on_image, run_yolo_on_image2

def Process_Start_Main(img, keypoints=None, out_dir="out"):
    """
    Stateless Verarbeitung:
    - Wenn keine Keypoints: Board wird automatisch erkannt.
    - Wenn Keypoints vorhanden: wird direkt die Dart-Erkennung gestartet.
    Rückgabe:
        darts: Liste [(x, y), ...]
        keypoints: {top, right, bottom, left}
    """

    try:
        SIZE = 400
        FLIP_X = False
        ANGLE_CORRECTION = 9
        os.makedirs(out_dir, exist_ok=True)

        print("📸 Starte Frame-Verarbeitung...")

        # ------------------------------------------------------------
        # 1️⃣ KEYPOINTS prüfen
        # ------------------------------------------------------------
        if keypoints is None or len(keypoints) < 4:
            print("⚠️ Keine gültigen Keypoints übergeben – erkenne Board mit YOLO...")
            # Lokaler Pfad für Entwicklung, Produktion: /opt/dartvision/models/Board.pt
            model_path = os.path.join(os.path.dirname(__file__), "models", "Board.pt")
            detected = run_yolo_on_image(model_path, img, wert=False)
            if len(detected) < 4:
                print("❌ Nicht genug Board-Keypoints erkannt.")
                return [], None

            keypoints = {
                "top": detected[0],
                "right": detected[1],
                "bottom": detected[2],
                "left": detected[3]
            }
            print("🟢 Board-Keypoints neu erkannt.")
        else:
            print("✅ Verwende übergebene Keypoints für Warping.")

        # ------------------------------------------------------------
        # 2️⃣ Homographie vorbereiten
        # ------------------------------------------------------------
        dst_pts = np.array([
            [SIZE / 2, 0],          # top
            [SIZE, SIZE / 2],       # right
            [SIZE / 2, SIZE],       # bottom
            [0, SIZE / 2]           # left
        ], dtype=np.float32)

        M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ANGLE_CORRECTION, 1.0)
        ones = np.ones((dst_pts.shape[0], 1))
        dst_hom = np.hstack([dst_pts, ones])
        dst_rot = (M @ dst_hom.T).T.astype(np.float32)

        src_pts = np.array([
            keypoints["top"],
            keypoints["right"],
            keypoints["bottom"],
            keypoints["left"]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src_pts, dst_rot)
        warped = cv2.warpPerspective(img, H, (SIZE, SIZE))

        # ------------------------------------------------------------
        # 3️⃣ DARTS erkennen (nur wenn Board-Keypoints existieren)
        # ------------------------------------------------------------
        print("🎯 Starte Dart-Erkennung...")
        # Lokaler Pfad für Entwicklung, Produktion: /opt/dartvision/models/Darts.pt
        dart_model_path = os.path.join(os.path.dirname(__file__), "models", "Darts.pt")
        dart_hits_raw = run_yolo_on_image2(
            dart_model_path,
            img,
            wert=False
        )

        if not dart_hits_raw:
            print("⚠️ Keine Darts erkannt.")
            return [], keypoints, {}

        # ------------------------------------------------------------
        # 4️⃣ DART-Koordinaten transformieren
        # ------------------------------------------------------------
        dart_hits = transform_dart_keypoints_absolute(
            dart_hits_raw, H, M, SIZE=SIZE, FLIP_X=FLIP_X
        )

        print(f"🎯 Gefundene Darts: {dart_hits}")

        # ------------------------------------------------------------
        # 5️⃣ Optional: Simulation / Visualisierung
        # ------------------------------------------------------------
        test_points = {f"Dart {i}": (int(x), int(y))
                       for i, (x, y) in enumerate(dart_hits, start=1)}
        if dart_hits:
            dart_scores = run_simulation(warped, out_dir, test_points=test_points)
        else:
            dart_scores ={}
        # ------------------------------------------------------------
        # 6️⃣ Rückgabe
        # ------------------------------------------------------------
        return dart_hits, keypoints, dart_scores

    except Exception as e:
        print(f"❌ Fehler in Process_Start_Main: {e}")
        return [], keypoints or None, {}
