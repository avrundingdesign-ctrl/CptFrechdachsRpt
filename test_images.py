#!/usr/bin/env python3
"""
Test-Skript zum Verarbeiten der Test-Bilder
"""
import os
import cv2
import json
import numpy as np
from PIL import Image
from V4_Warp_Image_keypoints import Process_Start_Main

# HEIC-Unterstützung aktivieren
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✅ HEIC-Unterstützung aktiviert")
except ImportError:
    print("⚠️ pillow-heif nicht installiert, HEIC-Bilder können nicht geladen werden")

def convert_heic_to_cv2(heic_path):
    """Konvertiert HEIC zu OpenCV-Format"""
    try:
        # Lade HEIC mit Pillow (nach register_heif_opener)
        img = Image.open(heic_path)
        # Konvertiere zu RGB falls nötig
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Konvertiere zu numpy array
        img_array = np.array(img)
        # OpenCV verwendet BGR statt RGB
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        print(f"❌ Fehler beim Laden von {heic_path}: {e}")
        return None

def test_image(image_path):
    """Testet ein einzelnes Bild"""
    print(f"\n{'='*60}")
    print(f"📸 Verarbeite: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Lade Bild
    if image_path.lower().endswith('.heic'):
        img = convert_heic_to_cv2(image_path)
    else:
        img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ Konnte Bild nicht laden: {image_path}")
        return None
    
    print(f"✅ Bild geladen: {img.shape[1]}x{img.shape[0]} Pixel")
    
    # Verarbeite Bild
    try:
        result = Process_Start_Main(img, keypoints=None)

        if isinstance(result, (list, tuple)) and len(result) == 4:
            darts, keypoints, dart_scores, detection_confidences = result
        else:
            darts, keypoints, dart_scores = result
            detection_confidences = {"board": {}, "darts": []}
        
        # Erstelle Response-Format wie im Server
        darts_with_scores = []
        dart_confidences = detection_confidences.get("darts", [])
        for i, coords in enumerate(darts, start=1):
            name = f"Dart {i}"
            x, y = coords
            dart_info = dart_scores.get(name, {})
            confidence = dart_confidences[i - 1] if i - 1 < len(dart_confidences) else None
            
            if isinstance(dart_info, dict):
                score = dart_info.get("score", 0)
                field_type = dart_info.get("field_type", "miss")
            else:
                score = dart_info if isinstance(dart_info, (int, float)) else 0
                field_type = "miss"
            
            darts_with_scores.append({
                "x": int(x),
                "y": int(y),
                "score": score,
                "field_type": field_type,
                "confidence": confidence
            })
        
        result = {
            "keypoints": keypoints,
            "keypoint_confidences": detection_confidences.get("board", {}),
            "darts": darts_with_scores
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Fehler bei der Verarbeitung: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(__file__), "Test")
    
    if not os.path.exists(test_dir):
        print(f"❌ Test-Verzeichnis nicht gefunden: {test_dir}")
        exit(1)
    
    # Finde alle Bilder
    image_files = []
    for ext in ['.heic', '.HEIC', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend([f for f in os.listdir(test_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"❌ Keine Bilder im Test-Verzeichnis gefunden")
        exit(1)
    
    print(f"🔍 Gefundene Bilder: {len(image_files)}")
    
    # Verarbeite alle Bilder
    all_results = {}
    for img_file in sorted(image_files):
        img_path = os.path.join(test_dir, img_file)
        result = test_image(img_path)
        if result:
            all_results[img_file] = result
    
    # Zeige Zusammenfassung
    print(f"\n{'='*60}")
    print("📊 ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    
    for img_file, result in all_results.items():
        print(f"\n📸 {img_file}:")
        print(f"   Keypoints: {result.get('keypoints', 'N/A')}")
        print(f"   Darts gefunden: {len(result.get('darts', []))}")
        for dart in result.get('darts', []):
            print(f"      - Position: ({dart['x']}, {dart['y']}), Score: {dart['score']}, Type: {dart['field_type']}")
    
    # Speichere JSON-Output
    output_file = os.path.join(test_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Ergebnisse gespeichert in: {output_file}")

