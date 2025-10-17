import cv2
import numpy as np
import math
import os

# ------------------ Board-Setup ------------------
SIZE = 400
R = SIZE // 2
CENTER = (R, R)

# Radien direkt in Pixeln für SIZE=400 (R=200)
R_INNER_BULL = 7.5
R_OUTER_BULL = 18.7
R_TRIPLE_IN  = 116.5
R_TRIPLE_OUT = 125.9
R_DOUBLE_IN  = 190.6
R_DOUBLE_OUT = 200.0   # Außenkante

# Sektoren im Uhrzeigersinn, 12 Uhr = 20
SECTORS = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]

# ------------------ Scoring ------------------
def get_score(x, y, sectors):
    dx, dy = x - R, y - R
    r = math.hypot(dx, dy)  # Abstand vom Mittelpunkt
    ang = (math.degrees(math.atan2(dx, -dy)) + 360) % 360  # Winkel 0°=oben, CW
    print(ang)
    if (x, y) == (0, 100):
        print(f"[DEBUG] x,y={(x,y)}  dx,dy={(dx,dy)}  r={r:.2f}  ang={ang:.2f}")
        print(f"[DEBUG] RADS  IB={R_INNER_BULL}  OB={R_OUTER_BULL}  "
              f"T=[{R_TRIPLE_IN},{R_TRIPLE_OUT}]  D=[{R_DOUBLE_IN},{R_DOUBLE_OUT}]")
    # Bulls
    if r <= R_INNER_BULL:
        return 50
    if r <= R_OUTER_BULL:
        return 25

    # passenden Sektor finden
    field = None
    for seg in sectors:
        start, end = seg["start"] % 360, seg["end"] % 360
        if start < end:
            inside = start <= ang < end
        else:  # Wraparound (z.B. 351°–9°)
            inside = ang >= start or ang < end
        if inside:
            field = seg["num"]
            break

    if field is None:
        return 0  # sollte nicht vorkommen

    # Ring bestimmen
    if R_TRIPLE_IN <= r <= R_TRIPLE_OUT:
        return field * 3
    if R_DOUBLE_IN <= r <= R_DOUBLE_OUT:
        return field * 2
    if r < R_DOUBLE_OUT:
        return field

    return 0  # außerhalb
def rot90_ccw(x, y, R):
    """90° gegen den Uhrzeigersinn um das Zentrum (R,R) drehen."""
    x_new = y
    y_new = 2*R - x
    return int(round(x_new)), int(round(y_new))

# ------------------ Overlay-Zeichnung ------------------
def run_simulation(warped_img, out_dir, test_points=None):
    overlay = warped_img.copy()
    sectors = []
    dart_scores = {}  # 🔹 Hier speichern wir nur die einzelnen Scores

    # ------------------ Ringe zeichnen ------------------
    for r_pix in [R_INNER_BULL, R_OUTER_BULL, R_TRIPLE_IN, R_TRIPLE_OUT, R_DOUBLE_IN, R_DOUBLE_OUT]:
        cv2.circle(overlay, CENTER, int(round(r_pix)), (0, 0, 0), 1)

    highlight = {3: (0, 0, 255), 6: (0, 255, 0), 11: (255, 0, 0), 20: (0, 255, 255)}
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, num in enumerate(SECTORS):
        angle_start = i * 18 - 9
        angle_end = (i + 1) * 18 - 9
        sectors.append({"num": num, "start": angle_start, "end": angle_end})

        if num in highlight:
            pts = []
            for a in np.linspace(angle_start, angle_end, 30):
                a_rad = math.radians(a)
                x = int(CENTER[0] + R * math.sin(a_rad))
                y = int(CENTER[1] - R * math.cos(a_rad))
                pts.append([x, y])
            pts.append([CENTER[0], CENTER[1]])
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], highlight[num])

        a_rad = math.radians(angle_start)
        x = int(CENTER[0] + R * math.sin(a_rad))
        y = int(CENTER[1] - R * math.cos(a_rad))
        cv2.line(overlay, CENTER, (x, y), (0, 0, 0), 1)

        a_mid = math.radians((angle_start + angle_end) * 0.5)
        x_txt = int(CENTER[0] + (R + 20) * math.sin(a_mid))
        y_txt = int(CENTER[1] - (R + 20) * math.cos(a_mid))
        cv2.putText(overlay, str(num), (x_txt - 10, y_txt + 5),
                    font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ------------------ Trefferpunkte ------------------
    if test_points:
        for name, (x, y) in test_points.items():
            score = get_score(x, y, sectors)
            dart_scores[name] = score  # 🔹 Nur hier speichern wir den Wert
            cv2.circle(overlay, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(overlay, f"{score}", (int(x) + 10, int(y)),
                        font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            print(f"{name:15s} -> Score = {score}")

    # ------------------ Overlay speichern ------------------
    out_path = os.path.join(out_dir, "warped_withBoard1.jpg")
    cv2.imwrite(out_path, overlay)
    print("[OK] Overlay gespeichert:", out_path)

    # 🔹 Nur die Einzelwerte zurückgeben
    return dart_scores


# ------------------ Selbsttest ------------------
if __name__ == "__main__":
    dummy = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 255

    # Beispiel-Testpunkte
    test_points = {
        "Bullseye": (R, R),
        "Triple 20": (R + int(R - (R_TRIPLE_IN + R_TRIPLE_OUT)/2), R),  # oberhalb Mitte
        "Double 20": (R, int(R - (R_DOUBLE_IN + R_DOUBLE_OUT)/2)),
        "Single 6": (int(R + (R_TRIPLE_OUT+R_DOUBLE_IN)/2), R),
    }

    run_simulation(dummy, ".", test_points=test_points)
