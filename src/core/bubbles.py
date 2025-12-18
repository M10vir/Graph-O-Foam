import numpy as np
import cv2

def detect_bubbles(img_u8: np.ndarray):
    if img_u8.ndim == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_u8, (0,0), 1.0)

    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=51, C=-2
    )

    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    H, W = img_u8.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 40 or area > (H*W*0.20):
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = float(4*np.pi*area/(peri*peri))

        (x, y), r = cv2.minEnclosingCircle(c)
        r = float(r)
        if r < 2 or r > 200:
            continue

        bubbles.append({
            "x": float(x), "y": float(y), "r": r,
            "area": float(area), "circularity": circ
        })

    return bubbles, th

def draw_overlay(img_u8: np.ndarray, bubbles, color=(0,255,0)):
    if img_u8.ndim == 2:
        out = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    else:
        out = img_u8.copy()
    for b in bubbles:
        x, y, r = int(b["x"]), int(b["y"]), int(b["r"])
        cv2.circle(out, (x,y), r, color, 1)
    return out

def bubble_stats(bubbles):
    if not bubbles:
        return {"n": 0, "r_mean": np.nan, "r_std": np.nan, "circ_mean": np.nan}
    rs = np.array([b["r"] for b in bubbles], dtype=np.float32)
    cs = np.array([b["circularity"] for b in bubbles], dtype=np.float32)
    return {
        "n": int(len(bubbles)),
        "r_mean": float(rs.mean()),
        "r_std": float(rs.std()),
        "circ_mean": float(np.nanmean(cs)),
    }
