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
        upload_folder = "/opt/dartvision/uploads"
        os.makedirs(upload_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = f"{timestamp}.jpg"
        save_path = os.path.join(upload_folder, filename)
        file.save(save_path)
        print(f"💾 Gespeichert: {save_path}")
        img = cv2.imread(save_path)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

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

        for i, coords in enumerate(darts, start=1):
            name = f"Dart {i}"
            x, y = coords
            score = dart_scores.get(name, 0)
            darts_with_scores.append({
                "x": x,
                "y": y,
                "score": score
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
