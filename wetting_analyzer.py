import cv2
import numpy as np
import pandas as pd
import re
import os
import sys
import argparse
import time
from collections import deque

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
POLY_DEG         = 2      # bậc polynomial fit
FIT_HEIGHT_FRAC  = 0.15   # % chiều cao giọt dùng để fit
MIN_DROP_AREA_FR = 0.003  # contour phải >= 0.3% diện tích frame
TEMP_SMOOTH_WIN  = 7      # cửa sổ moving-median để smooth nhiệt độ
ANGLE_SMOOTH_WIN = 5      # cửa sổ smooth góc (loại spike)
SAMPLE_INTERVAL  = 1.0    # giây giữa 2 lần ghi dữ liệu vào CSV
OCR_INTERVAL_FR  = 15     # OCR mỗi N frame (không cần OCR từng frame)


# ══════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════
_ocr_reader = None

def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            print("[OCR] Đang load EasyOCR...")
            _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[OCR] EasyOCR sẵn sàng.")
        except ImportError:
            print("[WARN] EasyOCR chưa cài (pip install easyocr). OCR bị tắt.")
    return _ocr_reader


def ocr_temperature(frame: np.ndarray) -> float | None:
    reader = get_ocr()
    if reader is None:
        return None

    h, w = frame.shape[:2]

    # Crop CỐ ĐỊNH góc trên trái: tối đa 200px rộng, 70px cao
    # (an toàn cho cả video 640×480 lẫn resolution khác)
    # từ:
    x2 = 120
    y2 = min(int(h * 0.12), 55)
    roi_bgr = frame[0:y2, 0:x2]
    if roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Thử tách chữ cyan bằng HSV mask trước
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
    mask_cyan  = cv2.inRange(hsv, (70, 30, 120), (130, 255, 255))
    mask_text  = cv2.bitwise_or(mask_white, mask_cyan)
    # Chỉ apply mask nếu có đủ pixel text
    if cv2.countNonZero(mask_text) > 300:
        gray = cv2.bitwise_and(gray, gray, mask=mask_text)

    # Upscale ×5 — chữ nhỏ (~30px cao) cần scale lớn để OCR tốt
    up = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # Chữ sáng trên nền tối → THRESH_BINARY với ngưỡng cứng 140
    _, tbin  = cv2.threshold(up, 140, 255, cv2.THRESH_BINARY)
    # Otsu thêm để fallback
    _, totsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for img in [tbin, totsu, cv2.bitwise_not(totsu)]:
        try:
            res = reader.readtext(img, detail=0, paragraph=False,
                                  allowlist='0123456789°ºCc. ')
            val = _parse_temp(" ".join(res))
            if val is not None:
                return val
        except Exception:
            pass
    return None


def _parse_temp(text: str) -> float | None:
    if not text:
        return None
    # Pattern chính: số + °C — ưu tiên cao nhất
    m = re.search(r'(\d{2,3})\s*[°ºoO]?\s*[Cc]', text)
    if m:
        v = float(m.group(1))
        if 20 <= v <= 400:
            return v
    # Fallback: chỉ chấp nhận 2 chữ số (tránh nhầm timestamp 3 chữ số)
    for n in re.findall(r'\b(\d{2})\b', text):
        v = float(n)
        if 20 <= v <= 99:
            return v
    return None  # KHÔNG fallback 3 chữ số nữa


# ══════════════════════════════════════════════════════════════
# BASELINE
# ══════════════════════════════════════════════════════════════
def auto_detect_baseline(frame: np.ndarray) -> int:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    y1_scan = int(h * 0.55)
    y2_scan = int(h * 0.92)
    row_means = np.mean(gray[y1_scan:y2_scan, :], axis=1)

    # Smooth nhẹ
    kernel = np.ones(5) / 5
    smooth = np.convolve(row_means, kernel, mode='same')

    # Tìm gradient lớn nhất (bước nhảy sáng đột ngột = mép trên substrate)
    grad = np.gradient(smooth)
    peak_idx = int(np.argmax(grad))
    baseline_y = peak_idx + y1_scan

    # Sanity check
    if int(h * 0.55) <= baseline_y <= int(h * 0.92):
        return baseline_y

    # Fallback Hough
    roi = frame[int(h * 0.50):int(h * 0.92)]
    edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 20, 70)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=int(w * 0.25),
                             minLineLength=int(w * 0.25),
                             maxLineGap=40)
    candidates = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(y2 - y1) <= 8:
                candidates.append((y1 + y2) / 2 + int(h * 0.50))
    if candidates:
        return int(np.median(sorted(candidates)[:3]))
    return int(h * 0.80)


def fit_baseline_from_clicks(p1, p2) -> tuple:
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) < 1:
        return 0.0, float(y1)
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def baseline_y_at(x: int, slope: float, intercept: float) -> int:
    return int(round(slope * x + intercept))


# ══════════════════════════════════════════════════════════════
# CONTOUR
# ══════════════════════════════════════════════════════════════
def best_channel(bgr: np.ndarray) -> np.ndarray:
    """Kênh màu có độ tương phản cao nhất (std lớn nhất)."""
    chs = cv2.split(bgr)
    return chs[int(np.argmax([np.std(c) for c in chs]))]


def find_drop_contour(frame: np.ndarray,
                      baseline_slope: float,
                      baseline_intercept: float):
    h, w = frame.shape[:2]
    y_top = int(h * 0.15)

    # Tạo mask loại trừ vùng dưới baseline (bao gồm baseline nghiêng)
    mask_above = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        yb = baseline_y_at(x, baseline_slope, baseline_intercept)
        top = max(y_top, 0)
        bot = max(min(yb - 2, h - 1), top)
        if bot > top:
            mask_above[top:bot, x] = 255

    crop_bgr = cv2.bitwise_and(frame, frame, mask=mask_above)
    crop_gray = best_channel(crop_bgr)

    # Chỉ lấy vùng có mask
    crop_only = crop_gray[y_top:, :]
    x1_roi = int(w * 0.10)
    x2_roi = int(w * 0.90)
    crop_only = crop_only[:, x1_roi:x2_roi]
    if crop_only.size == 0:
        return None, np.zeros((10, 10), np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop_only)
    blurred  = cv2.GaussianBlur(enhanced, (7, 7), 0)

    binary = cv2.Canny(blurred, 60, 160)
    binary = cv2.dilate(binary, None, iterations=1)  # giảm từ 2 xuống 1
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, binary

    crop_h, crop_w = crop_only.shape[:2]
    min_area = crop_h * crop_w * 0.05  # 5% vùng crop
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        # Contour phải có hình dạng hợp lý: không quá vuông vức
        x_c, y_c, w_c, h_c = cv2.boundingRect(c)
        aspect = w_c / (h_c + 1e-6)
        if aspect > 8:   # loại contour quá dẹt ngang (nhiễu nền)
            continue
        if y_c + h_c > crop_h * 0.5:   # phải chạm xuống nửa dưới vùng crop
            valid.append(c)

    if not valid:
        valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        return None, binary

    drop_c = max(valid, key=cv2.contourArea)
    # Dịch về tọa độ frame gốc
    drop_c_global = drop_c + np.array([[[x1_roi, y_top]]])
    return drop_c_global, binary


# ══════════════════════════════════════════════════════════════
# TÍNH GÓC
# ══════════════════════════════════════════════════════════════
def compute_contact_angles(drop_contour: np.ndarray,
                            baseline_slope: float,
                            baseline_intercept: float,
                            poly_deg: int = POLY_DEG,
                            fit_frac: float = FIT_HEIGHT_FRAC
                            ) -> tuple[float | None, float | None,
                                       tuple | None, tuple | None]:
    pts = drop_contour.reshape(-1, 2).astype(float)
    xs, ys = pts[:, 0], pts[:, 1]

    # Chiều cao giọt
    y_top_drop = ys.min()
    # Baseline y trung bình tại vùng giọt
    x_mid_drop = (xs.min() + xs.max()) / 2
    baseline_y_mid = baseline_y_at(int(x_mid_drop),
                                    baseline_slope, baseline_intercept)
    drop_height = baseline_y_mid - y_top_drop
    if drop_height < 15:
        return None, None, None, None

    fit_height_px = max(20, int(drop_height * fit_frac))

    # Tính khoảng cách mỗi điểm đến baseline nghiêng
    # dist > 0 = trên baseline (giọt); dist ≈ 0 = tại baseline
    def dist_to_baseline(px, py):
        # Baseline: slope*x - y + intercept = 0 (chuẩn hóa)
        # Dương = trên (trong tọa độ ảnh y↓ nghĩa là y < baseline_y)
        return (baseline_slope * px - py + baseline_intercept) / np.sqrt(baseline_slope**2 + 1)

    dists = np.array([dist_to_baseline(x, y) for x, y in zip(xs, ys)])

    # Điểm gần baseline: dist nhỏ (tức là ở sát đường baseline)
    # Lấy điểm có dist trong [0, fit_height_px] (phía trên baseline)
    near_mask = (dists >= -3) & (dists <= fit_height_px)
    if near_mask.sum() < 8:
        return None, None, None, None

    near_xs = xs[near_mask]
    near_ys = ys[near_mask]
    x_mid = (near_xs.min() + near_xs.max()) / 2

    angles, contact_pts = [], []
    for side, cond in [("left",  near_xs <= x_mid + 10),
                        ("right", near_xs >= x_mid - 10)]:
        sx, sy = near_xs[cond], near_ys[cond]
        if len(sx) < 6:
            angles.append(None)
            contact_pts.append(None)
            continue

        # Contact point: điểm có dist nhỏ nhất (gần baseline nhất)
        d_side = np.array([dist_to_baseline(x, y) for x, y in zip(sx, sy)])
        idx_cp = np.argmin(np.abs(d_side))
        cp_x, cp_y = sx[idx_cp], sy[idx_cp]

        # Cửa sổ fit phía trên contact point
        win = (sy >= cp_y - fit_height_px) & (sy <= cp_y + 5)
        sx_w, sy_w = sx[win], sy[win]
        if len(sx_w) < 5:
            angles.append(None)
            contact_pts.append(None)
            continue

        # Loại outlier IQR
        q1, q3 = np.percentile(sx_w, 20), np.percentile(sx_w, 80)
        iqr = q3 - q1 + 1e-6
        ok = (sx_w >= q1 - 2 * iqr) & (sx_w <= q3 + 2 * iqr)
        sx_w, sy_w = sx_w[ok], sy_w[ok]
        if len(sx_w) < 4:
            angles.append(None)
            contact_pts.append(None)
            continue

        try:
            pts_fit = np.column_stack([sx_w, sy_w]).astype(np.float32)
            vx, vy, _, _ = cv2.fitLine(pts_fit, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx), float(vy)
            # dxdy = vx/vy (dx/dy theo hướng tiếp tuyến)
            dxdy = vx / (vy + 1e-9)

            # Tiếp tuyến theo đường baseline nghiêng:
            # Vector tiếp tuyến giọt: (dxdy, 1) [vì y làm param]
            # Vector baseline: (1, slope)
            # Góc giữa 2 vector:
            tan_vec   = np.array([dxdy, 1.0])
            base_vec  = np.array([1.0, baseline_slope])

            cos_t = np.dot(tan_vec, base_vec) / (
                np.linalg.norm(tan_vec) * np.linalg.norm(base_vec) + 1e-9)
            cos_t = np.clip(cos_t, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(abs(cos_t)))

            # Sanity: reject nếu hướng tiếp tuyến ngược
            if side == "left"  and dxdy < -0.3:
                angles.append(None); contact_pts.append(None); continue
            if side == "right" and dxdy >  0.3:
                angles.append(None); contact_pts.append(None); continue

            angle_deg = float(np.clip(angle_deg, 3.0, 177.0))
            angles.append(round(angle_deg, 1))
            contact_pts.append((int(cp_x), int(cp_y)))

        except Exception:
            angles.append(None)
            contact_pts.append(None)

    return angles[0], angles[1], contact_pts[0], contact_pts[1]


def draw_tangent_line(img, cp, dxdy, baseline_slope, color, length=60):
    if cp is None:
        return
    cx, cy = cp
    # Hướng tiếp tuyến: (dxdy, 1) normalized
    norm = np.sqrt(dxdy**2 + 1.0)
    dx_n, dy_n = dxdy / norm, 1.0 / norm
    pt1 = (int(cx - dx_n * length), int(cy - dy_n * length))
    pt2 = (int(cx + dx_n * length), int(cy + dy_n * length))
    cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
# SMOOTHER
# ══════════════════════════════════════════════════════════════
class MedianSmoother:
    def __init__(self, window: int):
        self.window = window
        self.buf: deque = deque(maxlen=window)

    def push(self, val):
        if val is not None:
            self.buf.append(val)

    def get(self):
        if not self.buf:
            return None
        arr = np.array(self.buf)
        med = float(np.median(arr))
        # Loại outlier: giữ giá trị trong [med - 2σ, med + 2σ]
        std = np.std(arr) + 1e-6
        ok = arr[np.abs(arr - med) <= 2 * std]
        return float(np.median(ok)) if len(ok) else med


# ══════════════════════════════════════════════════════════════
# CLICK HANDLER
# ══════════════════════════════════════════════════════════════
class ClickCollector:
    def __init__(self, n: int = 2, label: str = "click"):
        self.n = n
        self.label = label
        self.pts: list = []
        self.done = False

    def reset(self):
        self.pts = []
        self.done = False

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.done:
            self.pts.append((x, y))
            print(f"  [{self.label}] Điểm {len(self.pts)}: ({x}, {y})")
            if len(self.pts) >= self.n:
                self.done = True


# ══════════════════════════════════════════════════════════════
# MAIN ANALYZER
# ══════════════════════════════════════════════════════════════
def run(video_path: str, output_dir: str,
        sample_interval: float = SAMPLE_INTERVAL,
        show_binary: bool = True):

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "wetting_angles.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_fr / fps
    print(f"[INFO] Video: {os.path.basename(video_path)}")
    print(f"[INFO] FPS={fps:.1f} | Frames={total_fr} | Thời lượng={duration/60:.1f} phút")
    print(f"[INFO] Ghi CSV mỗi {sample_interval}s → ~{int(duration/sample_interval)} điểm")

    WIN = "Wetting Angle Analyzer"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 900, 600)

    # ── Đọc frame đầu ────────────────────────────────────────
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Không đọc được frame đầu.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    h_fr, w_fr = first_frame.shape[:2]

    # ── Baseline setup ────────────────────────────────────────
    baseline_slope     = 0.0
    baseline_intercept = float(auto_detect_baseline(first_frame))
    baseline_ready     = True
    print(f"[INFO] Auto-detect baseline_y ≈ {int(baseline_intercept)}")
    print("[HINT] Bấm [B] để set baseline thủ công (click 2 điểm)")
    print("[HINT] [A]=auto-detect  [P]=pause  [S]=step  [R]=reset baseline  [Q]=thoát+lưu")

    collector = ClickCollector(n=2, label="baseline")
    waiting_for_baseline = False

    def mouse_cb(event, x, y, flags, param):
        if waiting_for_baseline:
            collector.callback(event, x, y, flags, param)

    cv2.setMouseCallback(WIN, mouse_cb)

    # ── State ─────────────────────────────────────────────────
    paused        = False
    speed_mult    = 1.0   # 0.25 … 4.0
    temp_smoother = MedianSmoother(TEMP_SMOOTH_WIN)
    al_smoother   = MedianSmoother(ANGLE_SMOOTH_WIN)
    ar_smoother   = MedianSmoother(ANGLE_SMOOTH_WIN)
    last_temp     = None
    last_save_ts  = -999.0
    frame_idx     = 0
    ocr_counter   = 0
    records       = []

    # ── Vòng lặp chính ───────────────────────────────────────
    while True:
        # ---- PAUSE / STEP ----------------------------------------
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'): break
            if key == ord('p'): paused = False
            if key == ord('s'):
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                # process single frame (fall through bên dưới)
            else:
                continue

        # ---- ĐỌC FRAME -------------------------------------------
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

        timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # ---- BASELINE MODE: chờ click ----------------------------
        if waiting_for_baseline:
            disp = frame.copy()
            for pp in collector.pts:
                cv2.circle(disp, pp, 6, (0, 255, 255), -1)
            cv2.putText(disp,
                        f"Click điểm {len(collector.pts)+1}/2 trên baseline",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
            cv2.imshow(WIN, disp)
            cv2.waitKey(30)
            if collector.done:
                baseline_slope, baseline_intercept = fit_baseline_from_clicks(
                    collector.pts[0], collector.pts[1])
                baseline_ready = True
                waiting_for_baseline = False
                print(f"[OK] Baseline set: slope={baseline_slope:.4f}  "
                      f"intercept={baseline_intercept:.1f}")
            continue

        # ---- OCR nhiệt độ (không phải mỗi frame) -----------------
        ocr_counter += 1
        if ocr_counter % OCR_INTERVAL_FR == 1:
            t = ocr_temperature(frame)
            if t is not None:
                temp_smoother.push(t)
                last_temp = t
        cur_temp = temp_smoother.get() or last_temp

        # ---- DETECT CONTOUR + TÍNH GÓC ---------------------------
        angle_left = angle_right = angle_mean = None
        cp_left = cp_right = None
        drop_contour = None
        binary_vis = None

        if baseline_ready:
            drop_contour, binary_vis = find_drop_contour(
                frame, baseline_slope, baseline_intercept)

            if drop_contour is not None:
                al, ar, cp_left, cp_right = compute_contact_angles(
                    drop_contour, baseline_slope, baseline_intercept)

                al_smoother.push(al)
                ar_smoother.push(ar)
                angle_left  = al_smoother.get()
                angle_right = ar_smoother.get()

                if angle_left is not None and angle_right is not None:
                    angle_mean = round((angle_left + angle_right) / 2, 1)
                elif angle_left is not None:
                    angle_mean = angle_left
                elif angle_right is not None:
                    angle_mean = angle_right

        # ---- GHI CSV theo interval --------------------------------
        if timestamp_s - last_save_ts >= sample_interval:
            records.append(dict(
                timestamp_s   = round(timestamp_s, 2),
                frame         = frame_idx,
                temperature_C = round(cur_temp, 1) if cur_temp else None,
                angle_left    = round(angle_left,  1) if angle_left  else None,
                angle_right   = round(angle_right, 1) if angle_right else None,
                angle_mean    = round(angle_mean,  1) if angle_mean  else None,
                baseline_slope     = round(baseline_slope, 5),
                baseline_intercept = round(baseline_intercept, 1),
            ))
            last_save_ts = timestamp_s

        # ── VẼ OVERLAY ────────────────────────────────────────────
        vis = frame.copy()

        # Baseline (đường đỏ)
        if baseline_ready:
            pt1 = (0, baseline_y_at(0, baseline_slope, baseline_intercept))
            pt2 = (w_fr, baseline_y_at(w_fr, baseline_slope, baseline_intercept))
            cv2.line(vis, pt1, pt2, (0, 0, 255), 2)

        # Contour giọt (xanh lá)
        if drop_contour is not None:
            cv2.drawContours(vis, [drop_contour], -1, (0, 220, 0), 2)

        # Contact points + arc hiển thị góc
        if cp_left is not None:
            cv2.circle(vis, cp_left, 6, (0, 255, 255), -1)
        if cp_right is not None:
            cv2.circle(vis, cp_right, 6, (255, 0, 255), -1)

        # Thanh info trên
        bar_h = 64
        bar   = np.zeros((bar_h, w_fr, 3), dtype=np.uint8)
        temp_str  = f"{cur_temp:.0f}°C" if cur_temp else "?°C"
        al_str    = f"L:{angle_left:.1f}°" if angle_left  else "L:--"
        ar_str    = f"R:{angle_right:.1f}°" if angle_right else "R:--"
        mean_str  = f"θ={angle_mean:.1f}°" if angle_mean else "θ=--"
        ts_str    = f"t={timestamp_s:.1f}s  fr={frame_idx}/{total_fr}"
        spd_str   = f"×{speed_mult:.2g}"
        st_str    = "PAUSE" if paused else "▶"

        cv2.putText(bar, f"T={temp_str}  {al_str}  {ar_str}  {mean_str}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
        cv2.putText(bar, f"{ts_str}   {spd_str}  {st_str}   [B]baseline [↑↓]adjust [P]pause [Q]quit",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,180,180), 1)

        combined = np.vstack([bar, vis])

        # Binary thumbnail (góc dưới phải)
        if show_binary and binary_vis is not None:
            brgb = cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR)
            sc   = min(0.22, 200 / max(binary_vis.shape))
            bsmall = cv2.resize(brgb, None, fx=sc, fy=sc)
            bh2, bw2 = bsmall.shape[:2]
            combined[-bh2:, -bw2:] = bsmall

        cv2.imshow(WIN, combined)

        # ---- KEY ------------------------------------------------
        delay = max(1, int(1000 / fps / speed_mult))
        key   = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("PAUSE" if paused else "RESUME")
        elif key == ord('s') and not paused:
            paused = True
        elif key == ord('b'):
            print("\n[Baseline] Click 2 điểm trên baseline trong cửa sổ video...")
            collector.reset()
            waiting_for_baseline = True
        elif key == ord('a'):
            baseline_slope = 0.0
            baseline_intercept = float(auto_detect_baseline(frame))
            baseline_ready = True
            print(f"[OK] Auto-detect baseline_y = {int(baseline_intercept)}")
        elif key == ord('r'):
            baseline_ready = False
            print("[RESET] Baseline đã xóa. Bấm [A] hoặc [B] để set lại.")
        elif key in (ord('+'), ord('=')):
            speed_mult = min(speed_mult * 2, 8.0)
        elif key == ord('-'):
            speed_mult = max(speed_mult / 2, 0.25)
        # Mũi tên lên/xuống: dịch baseline ±1px (giữ slope)
        elif key == 82:  # Arrow Up
            baseline_intercept -= 1
        elif key == 84:  # Arrow Down
            baseline_intercept += 1
        # Shift+mũi tên: dịch ±5px nhanh hơn
        elif key == 56:  # numpad 8
            baseline_intercept -= 5
        elif key == 50:  # numpad 2
            baseline_intercept += 5

    cap.release()
    cv2.destroyAllWindows()

    # ── Xuất CSV ─────────────────────────────────────────────
    df = pd.DataFrame(records)
    if not df.empty:
        # Nội suy nhiệt độ bị thiếu
        df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce")
        df["temperature_C"] = (df["temperature_C"]
                               .interpolate(method="linear", limit_direction="both")
                               .ffill().bfill())
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] CSV đã lưu: {csv_path}")
        print(f"     {len(df)} điểm dữ liệu")
        print(df[["timestamp_s", "temperature_C",
                   "angle_left", "angle_right", "angle_mean"]].tail(10).to_string(index=False))
    else:
        print("[WARN] Không có dữ liệu nào được ghi.")

    return df


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wetting Angle Analyzer — đo góc ướt trực tiếp từ video")
    parser.add_argument("--video",    required=True,
                        help="Đường dẫn file video (.mpg/.mp4/.avi)")
    parser.add_argument("--output",   default="output",
                        help="Thư mục output (mặc định: output/)")
    parser.add_argument("--interval", type=float, default=SAMPLE_INTERVAL,
                        help=f"Giây giữa 2 lần ghi CSV (mặc định: {SAMPLE_INTERVAL})")
    parser.add_argument("--no-binary", action="store_true",
                        help="Tắt thumbnail binary debug")
    args = parser.parse_args()

    run(video_path    = args.video,
        output_dir    = args.output,
        sample_interval = args.interval,
        show_binary   = not args.no_binary)