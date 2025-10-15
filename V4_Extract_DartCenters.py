import numpy as np
import cv2

def polygon_centroid(pts):
    x = pts[:,0]; y = pts[:,1]
    x2 = np.roll(x, -1); y2 = np.roll(y, -1)
    cross = x*y2 - x2*y
    A = cross.sum()/2.0
    if abs(A) < 1e-8:
        return np.array([x.mean(), y.mean()], dtype=np.float32)
    Cx = ((x + x2) * cross).sum() / (6*A)
    Cy = ((y + y2) * cross).sum() / (6*A)
    return np.array([Cx, Cy], dtype=np.float32)



def extract_dart_keypoints(label_path, img_width, img_height, H, M, SIZE=400, FLIP_X=False, dart_class=0, debug=False):
    """
    YOLO Keypoints: BBox (bildrelativ) + KP (bbox-relativ) -> KP in Bildpixel -> H -> Flip -> M -> (x,y)
    ACHTUNG: kx,ky sind relativ zur BBox (0..1), nicht zum Bild!
    """
    dart_points = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            cls = int(parts[0])
            if cls != dart_class:
                continue

            # BBox relativ zum Bild
            bx = float(parts[1]); by = float(parts[2])
            bw = float(parts[3]); bh = float(parts[4])

            # Erster Keypoint (bbox-relativ)
# Erster Keypoint (bbox-relativ)
            kx_rel = float(parts[5])
            ky_rel = float(parts[6])

            # Sichtbarkeit / Confidence
            vis = 2
            if len(parts) > 7:
                vis_raw = float(parts[7])
                vis = 0 if vis_raw <= 0 else 2

            if vis == 0:
                if debug:
                    print(f"[KP] cls={cls} -> vis=0 (übersprungen)")
                continue


            # --- WICHTIG: bbox-relativ -> Bildpixel ---
            kx_abs = (bx - bw/2.0 + kx_rel * bw) * img_width
            ky_abs = (by - bh/2.0 + ky_rel * bh) * img_height

            pt = np.array([[[kx_abs, ky_abs]]], dtype=np.float32)

            # Schritt 1: Homographie
            pt_warped = cv2.perspectiveTransform(pt, H)[0][0]

            # Schritt 2: Flip
            if FLIP_X:
                pt_warped[0] = (SIZE - 1) - pt_warped[0]

            # Schritt 3: Rotation (Angle Correction)
            # Schritt 3: Rotation (Angle Correction)
            pt_aligned = np.hstack([pt_warped, [1]]) @ M.T

            # Schritt 4: zusätzliche Korrektur um 189°
            EXTRA_ROT = 9 + 180  # Grad
            if EXTRA_ROT != 0:
                cx, cy = SIZE/2, SIZE/2
                dx, dy = pt_aligned[0] - cx, pt_aligned[1] - cy
                rad = np.radians(EXTRA_ROT)
                rot_x = dx*np.cos(rad) - dy*np.sin(rad)
                rot_y = dx*np.sin(rad) + dy*np.cos(rad)
                pt_aligned[0] = cx + rot_x
                pt_aligned[1] = cy + rot_y

            # Ergebnis speichern
            x_i, y_i = int(round(pt_aligned[0])), int(round(pt_aligned[1]))
            dart_points.append((x_i, y_i))

        

 
    return dart_points

def transform_dart_keypoints_absolute(dart_points_abs, H, M, SIZE=400, FLIP_X=False, extra_rot=9, debug=False):
    """
    Nimmt absolute Keypoints (Pixelkoordinaten im Originalbild),
    und transformiert sie ins normierte Board.
    """
    dart_points = []

    for (kx_abs, ky_abs) in dart_points_abs:
        pt = np.array([[[kx_abs, ky_abs]]], dtype=np.float32)

        # Schritt 1: Homographie
        pt_warped = cv2.perspectiveTransform(pt, H)[0][0]

        # Schritt 2: Flip
        if FLIP_X:
            pt_warped[0] = (SIZE - 1) - pt_warped[0]

        # Schritt 3: Rotation (Angle Correction)
        pt_aligned = np.hstack([pt_warped, [1]]) @ M.T

        # Schritt 4: zusätzliche Korrektur
        if extra_rot != 0:
            cx, cy = SIZE/2, SIZE/2
            dx, dy = pt_aligned[0] - cx, pt_aligned[1] - cy
            rad = np.radians(extra_rot)
            rot_x = dx*np.cos(rad) - dy*np.sin(rad)
            rot_y = dx*np.sin(rad) + dy*np.cos(rad)
            pt_aligned[0] = cx + rot_x
            pt_aligned[1] = cy + rot_y

        x_i, y_i = int(round(pt_aligned[0])), int(round(pt_aligned[1]))
        dart_points.append((x_i, y_i))

    if debug:
        print(f"[ABS] return points: {dart_points}")

    return dart_points

