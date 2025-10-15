import numpy as np
import os, json, numpy as np, cv2
from flask import Flask, request, jsonify
from V4_Warp_Image_keypoints import Process_Start_Main


app = Flask(__name__)

REF_FILE = r"UsefulScripts\WichtigeScripts\Aktuell\Versuch4\keys.json"

@app.route("/upload", methods=["POST"])
def upload():
    # Prüfen, ob Reset-Flag gesendet wurde
    if "reset" in request.form and request.form["reset"].lower() == "true":
        if os.path.exists(REF_FILE):
            os.remove(REF_FILE)

            return jsonify({"status": "reset_done"})
        else:

            return jsonify({"status": "no_ref_file"}), 404

    # Wenn kein Reset, dann normales Bild erwarten
    if "file" not in request.files:

        return jsonify({"error": "no file"}), 400


    file = request.files["file"]

    # Datei in Bytes umwandeln
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Debug
    if img is None:
        print("⚠️ cv2 konnte das Bild nicht dekodieren.")
        return jsonify({"error": "invalid image"}), 400

    # Haupt-Processing
    score = Process_Start_Main(img, out_dir="uploads")

    print(f"🏹 Score berechnet: {score}")
    return jsonify({"score": score})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
