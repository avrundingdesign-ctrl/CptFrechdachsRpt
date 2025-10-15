import os, json, numpy as np


def update_dart_json(dart_hits):
    DART_FILE = "/opt/dartvision/jsons/darts.json"
    DIST_THRESHOLD = 50  # Pixel Abstand, ab wann ein Dart als "neu" gilt
    MAX_DARTS = 3
    """
    Aktualisiert oder erstellt die darts.json.
    Fügt neue Darts hinzu, wenn sie weiter als DIST_THRESHOLD px von alten entfernt sind.
    Gibt zurück:
        (saved_darts, should_calculate_score)
    """
    # Datei anlegen, falls sie fehlt
    if not os.path.exists(DART_FILE):
        with open(DART_FILE, "w") as f:
            json.dump({"darts": []}, f)

    # Bestehende Darts laden
    with open(DART_FILE, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"darts": []}

    saved = data.get("darts", [])

    if not dart_hits:
        if saved:
            print("keine Dartserkannt - Darts.json wird zurückgesetzt")
            with open (DART_FILE, "w") as f:
                json.dump({"darts": []}, f)
        return [], False

    # Neue Darts prüfen & ggf. hinzufügen
    for (x, y) in dart_hits:
        is_new = True
        for old in saved:
            dist = np.linalg.norm(np.array([x, y]) - np.array([old["x"], old["y"]]))
            if dist < DIST_THRESHOLD:
                is_new = False
                break
        if is_new and len(saved) < MAX_DARTS:
            saved.append({"x": float(x), "y": float(y)})
            print(f"🎯 Neuer Dart erkannt: ({x:.1f}, {y:.1f})")

    # Datei aktualisieren
    with open(DART_FILE, "w") as f:
        json.dump({"darts": saved}, f, indent=2)

    # Check: Sind 3 Darts voll?
    if len(saved) < MAX_DARTS:
        print(f"⏳ {len(saved)}/3 Darts – warte auf weitere Würfe ...")
        return saved, False

    # Wenn 3 Darts da → Score berechnen und Datei zurücksetzen
    print("✅ 3 Darts erkannt – Score wird berechnet.")
    with open(DART_FILE, "w") as f:
        json.dump({"darts": []}, f)

    return saved, True
