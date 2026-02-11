from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np, cv2, json
from V4_Warp_Image_keypoints import Process_Start_Main
import os
import random
app = Flask(__name__)

# ============================================
# 🎭 FAKE MODE CONFIGURATION
# ============================================
# Setze FAKE_MODE = True, um Bilderkennung zu umgehen
# und vordefinierte Werte zurückzugeben
FAKE_MODE = True

x = random.randint(300,900),
y = random.randint(300, 900),

# Vordefinierte Fake-Antworten (kannst du hier anpassen)
FAKE_RESPONSES = {

    "triple_20": {
        "keypoints": {
            "top": [489.4, 472.0],
            "right": [961.2, 801.2],
            "bottom": [620.1, 1188.1],
            "left": [231.9, 924.9]
        },
        "darts": [
            {"x": 640, "y": 360, "score": 60, "field_type": "triple"},
            {"x": 638, "y": 335, "score": 40, "field_type": "triple"},
            {"x": 642, "y": 358, "score": 60, "field_type": "triple"}
        ]
    },
    "bullseye": {
        "keypoints": {
            "top": [489.4, 472.0],
            "right": [961.2, 801.2],
            "bottom": [620.1, 1188.1],
            "left": [231.9, 924.9]
        },
        "darts": [
            {"x": 640, "y": 640, "score": 50, "field_type": "bullseye"},
            {"x": 641, "y": 639, "score": 50, "field_type": "bullseye"}
        ]
    },
    "mixed": {
        "keypoints": {
            "top": [489.4, 472.0],
            "right": [961.2, 801.2],
            "bottom": [620.1, 1188.1],
            "left": [231.9, 924.9]
        },
        "darts": [
            {"x": 640, "y": 360, "score": 20, "field_type": "single"},
           # {"x": 720, "y": 450, "score": 18, "field_type": "double"},
           # {"x": 560, "y": 500, "score": 15, "field_type": "triple"}
        ]
    },
    "single_dart_random": {
        "keypoints": {
            "top": [489.4, 472.0],
            "right": [961.2, 801.2],
            "bottom": [620.1, 1188.1],
            "left": [231.9, 924.9]
        },
        "darts": [
            {"x": x, "y": y, "score": 19, "field_type": "single"}
        ]
    },
        "single_dart": {
        "keypoints": {
            "top": [489.4, 472.0],
            "right": [961.2, 801.2],
            "bottom": [620.1, 1188.1],
            "left": [231.9, 924.9]
        },
        "darts": [
            {"x": 30, "y": 70, "score": 40, "field_type": "tripple"}
        ]
    },
    "no_darts": {
        "keypoints": {
            "top": [489.4, 472.0],
            "right": [961.2, 801.2],
            "bottom": [620.1, 1188.1],
            "left": [231.9, 924.9]
        },
        "darts": []
    }
}

# Wähle hier, welche Fake-Antwort zurückgegeben wird
# Optionen: "triple_20", "bullseye", "mixed", "single_dart", "no_darts"
ACTIVE_FAKE_RESPONSE = "single_dart"


@app.route("/upload", methods=["POST"])
def upload():
    # ============================================
    # 🎭 FAKE MODE: Sende vordefinierte Antwort
    # ============================================
    if FAKE_MODE:
        fake_response = FAKE_RESPONSES.get(ACTIVE_FAKE_RESPONSE, FAKE_RESPONSES["triple_20"])
        print(f"🎭 FAKE MODE aktiv - Sende Antwort: {ACTIVE_FAKE_RESPONSE}")
        print(json.dumps(fake_response, indent=2))
        return jsonify(fake_response)
    
    # ============================================
    # 📸 NORMAL MODE: Echte Bilderkennung
    # ============================================
    try:
        global np

        # ------------------------------------------------
        # 1️⃣ Keypoints vom Client empfangen (optional)
        # ------------------------------------------------
        kp_json = request.form.get("keypoints")
        keypoints = json.loads(kp_json) if kp_json else None

        # ------------------------------------------------
        # 2️⃣ Bild empfangen
        # ------------------------------------------------
        if "file" not in request.files:
            return jsonify({"error": "Missing image file"}), 400

        file = request.files["file"]

        # 🔹 Datei direkt aus dem Speicher lesen
        file_bytes = np.frombuffer(file.read(), np.uint8)

        # 🔹 In ein OpenCV-Bild dekodieren
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        print(f"📸 Bild empfangen ({len(file_bytes) / 1024:.1f} KB)")

        # 📁 Original-Bild speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        os.makedirs("uploads", exist_ok=True)
        original_path = f"uploads/original_{timestamp}.jpg"
        cv2.imwrite(original_path, img)
        print(f"💾 Original-Bild gespeichert: {original_path}")

        # ------------------------------------------------
        # 3️⃣ Hauptverarbeitung starten
        # ------------------------------------------------
        try:
            darts, keypoints, dart_scores = Process_Start_Main(img, keypoints=keypoints)
        except ValueError as e:
            print(f"❌ Fehler beim Entpacken: {e}")

            # Funktion nochmal aufrufen, um zu sehen, was wirklich zurückkam
            result = Process_Start_Main(img, keypoints=keypoints)
            print("🔍 Tatsächliche Rückgabe von Process_Start_Main:", result)
            if isinstance(result, (list, tuple)):
                print("📦 Typen:", [type(r) for r in result])
                print(f"📊 Anzahl Rückgabewerte: {len(result)}")

                # Fallback, um Absturz zu vermeiden
                if len(result) == 2:
                    darts, keypoints = result
                    dart_scores = {}
                elif len(result) == 1:
                    darts = result[0]
                    keypoints = {}
                    dart_scores = {}
                else:
                    darts, keypoints, dart_scores = [], {}, {}
            else:
                print("⚠️ Rückgabewert ist kein Tuple oder List:", type(result))
                darts, keypoints, dart_scores = [], {}, {}

        # ------------------------------------------------
        # 4️⃣ Finales Dict (Response) zusammenbauen
        # ------------------------------------------------
        darts_with_scores = []
        
        # Sicherstellen, dass dart_scores ein Dictionary ist
        if dart_scores is None:
            dart_scores = {}

        for i, coords in enumerate(darts, start=1):
            name = f"Dart {i}"
            x, y = coords
            dart_info = dart_scores.get(name, {})
            
            # Unterstützung für alte Format (nur score als Zahl) und neues Format (dict mit score und field_type)
            if isinstance(dart_info, dict):
                score = dart_info.get("score", 0)
                field_type = dart_info.get("field_type", "miss")
            else:
                # Fallback für alte Format
                score = dart_info if isinstance(dart_info, (int, float)) else 0
                field_type = "miss"
            
            darts_with_scores.append({
                "x": x,
                "y": y,
                "score": score,
                "field_type": field_type
            })

        response = {
            "keypoints": keypoints,
            "darts": darts_with_scores
        }
        import numpy as np

        

        print("🔍 --- Typanalyse der Response ---")
        describe_types({
            "keypoints": keypoints,
            "darts": darts_with_scores
        })
        print("🔍 --- Ende der Typanalyse ---")
        
        print("server resposne")
        print(json.dumps(response, indent=2))

        return jsonify(response)

    except Exception as e:
        print(f"❌ Fehler im Upload: {e}")
        return jsonify({"error": str(e)}), 500
    
def describe_types(obj, name="root", indent=0):
            prefix = " " * indent
            if isinstance(obj, dict):
                print(f"{prefix}🧩 {name} (dict):")
                for k, v in obj.items():
                    describe_types(v, name=f"{k}", indent=indent + 2)
            elif isinstance(obj, list):
                print(f"{prefix}📜 {name} (list, len={len(obj)}):")
                for i, v in enumerate(obj[:3]):  # nur die ersten paar anzeigen
                    describe_types(v, name=f"[{i}]", indent=indent + 2)
            else:
                print(f"{prefix}🔹 {name}: {type(obj)} -> {repr(obj)}")

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)      
