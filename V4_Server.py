from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np, cv2, json
from V4_Warp_Image_keypoints import Process_Start_Main
import os
app = Flask(__name__)



@app.route("/upload", methods=["POST"])
def upload():
    #print("🧪 Sende direkt Fake-Antwort (kein echtes Processing).")

    #fake_response = {
        #"keypoints": {
            #"top": [489.4, 472.0],
            #"right": [961.2, 801.2],
           # "bottom": [620.1, 1188.1],
           # "left": [231.9, 924.9]
     #   },
        #"darts": [
           # {"x": 191, "y": 128, "score": 20},
            #{"x": 309, "y": 177, "score": 13},
           # {"x": 227, "y": 260, "score": 17}
       # ]
    #}

    #return jsonify(fake_response)
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
    import os
    # Lokaler Entwicklungsserver
    if os.getenv("ENV") == "production":
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        app.run(host="127.0.0.1", port=5000, debug=True)
