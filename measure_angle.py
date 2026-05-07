"""
measure_angle.py  (v3 — robust pipeline)
-----------------------------------------
Pipeline:
  Video frame
    → Crop ROI  (bỏ text overlay + vùng sáng substrate)
    → Tách kênh màu tốt nhất (thường kênh Blue vì giọt kim loại xanh tím)
    → CLAHE + adaptive threshold
    → Morphological fill
    → Canny edge
    → Detect baseline  (HoughLinesP + lọc chặt + fallback scan)
    → Tìm contour giọt lớn nhất (area filter + position filter)
    → Xác định 2 contact point (giao điểm contour & baseline)
    → Fit polynomial bậc 2 cục bộ tại mỗi contact point
    → Tính tangent → contact angle

OCR:
  - Crop chính xác góc trên trái (text "133°C")
  - Scale lên lớn, threshold chặt cho chữ trắng trên nền tối
  - Regex fallback: 3 chữ số trong [50, 450]
  - Nội suy từ frame liền kề nếu OCR thất bại

CSV: [filename, timestamp_s, temperature_C, angle_left, angle_right, angle_mean, baseline_y, note]
"""

import cv2
import numpy as np
import os
import re
import pandas as pd
import sys

# ── EasyOCR lazy load ────────────────────────────────────────────────────────
_ocr_reader = None

def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except ImportError:
            print("[WARN] easyocr không cài, OCR sẽ dùng fallback.")
    return _ocr_reader


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OCR NHIỆT ĐỘ
# ═══════════════════════════════════════════════════════════════════════════════
def _preprocess_ocr_roi(roi_gray: np.ndarray) -> list:
    """
    Sinh ra nhiều biến thể tiền xử lý từ ROI grayscale (chữ trắng trên nền tối).
    Upscale ×4 để OCR chính xác hơn.
    """
    up = cv2.resize(roi_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    variants = []

    # A: ảnh gốc (chữ trắng)
    variants.append(up)

    # B: threshold 180 (giữ chữ sáng)
    _, t = cv2.threshold(up, 180, 255, cv2.THRESH_BINARY)
    variants.append(t)
    variants.append(cv2.bitwise_not(t))  # chữ tối trên nền trắng

    # C: Otsu
    _, ot = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(ot)
    variants.append(cv2.bitwise_not(ot))

    # D: morphological dilate (nối chữ)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    variants.append(cv2.dilate(t, k))

    return variants


def ocr_temperature(frame: np.ndarray) -> float | None:
    """
    Đọc nhiệt độ từ text overlay góc trên trái.
    Ảnh mẫu: "133°C" ở vị trí khoảng (x=0..160, y=0..40) trong frame 640×480.
    """
    h, w = frame.shape[:2]

    # ROI chính xác hơn — chữ nằm ở ~15% chiều cao, ~30% chiều rộng
    rois_coords = [
        (0, int(h * 0.14), 0, int(w * 0.28)),   # nhỏ chính xác
        (0, int(h * 0.18), 0, int(w * 0.38)),   # rộng hơn nếu cỡ chữ lớn
        (0, int(h * 0.22), 0, int(w * 0.50)),   # rộng nhất
    ]

    reader = get_ocr()

    for (y1, y2, x1, x2) in rois_coords:
        roi_bgr = frame[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            continue
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        variants = _preprocess_ocr_roi(roi_gray)

        for img in variants:
            if reader is not None:
                try:
                    results = reader.readtext(
                        img, detail=0, paragraph=False,
                        allowlist='0123456789°ºCc. '
                    )
                    text = " ".join(results)
                    val = _parse_temperature(text)
                    if val is not None:
                        return val
                except Exception:
                    pass

    return None


def _parse_temperature(text: str) -> float | None:
    if not text:
        return None
    # Pattern chính: số + °C (hoặc biến thể)
    match = re.search(r'(\d{2,4})\s*[°ºoO0]?\s*[Cc]', text)
    if match:
        val = float(match.group(1))
        if 50 <= val <= 450:
            return val
    # Fallback: 3 chữ số
    for n in re.findall(r'\b(\d{3})\b', text):
        val = float(n)
        if 50 <= val <= 450:
            return val
    # Fallback: 2 chữ số
    for n in re.findall(r'\b(\d{2})\b', text):
        val = float(n)
        if 50 <= val <= 99:
            return val
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TÁCH KÊNH MÀU TỐT NHẤT
# ═══════════════════════════════════════════════════════════════════════════════
def best_channel(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Chọn kênh màu có contrast giọt-nền cao nhất.
    Giọt kim loại thường tối nhất ở kênh Red hoặc Green.
    Dùng standard deviation làm proxy cho contrast.
    """
    b, g, r = cv2.split(crop_bgr)
    stds = [np.std(c) for c in [b, g, r]]
    best = np.argmax(stds)
    return [b, g, r][best]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DETECT BASELINE
# ═══════════════════════════════════════════════════════════════════════════════
def find_baseline_y(frame: np.ndarray) -> int:
    """
    Tìm y của đường substrate (đường ngang sáng mạnh ở vùng dưới).

    Strategy:
      1. Tìm vùng ROI từ 50% đến 95% chiều cao
      2. Dùng HoughLinesP để tìm đường ngang
      3. Nếu thất bại: scan từng hàng pixel, lấy hàng sáng nhất (mean intensity)
      4. Loại bỏ đường quá gần đáy (<3% margin)
    """
    h, w = frame.shape[:2]
    roi_y1 = int(h * 0.48)
    roi_y2 = int(h * 0.94)
    roi = frame[roi_y1:roi_y2, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Canny với ngưỡng thấp để bắt đường sáng rõ
    edges = cv2.Canny(gray, 20, 80, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=int(w * 0.35),
        minLineLength=int(w * 0.35),
        maxLineGap=30
    )

    candidates = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) <= 8:   # gần nằm ngang
                y_global = (y1 + y2) / 2 + roi_y1
                if y_global < h * 0.93:   # không quá sát đáy
                    candidates.append(y_global)

    if candidates:
        # Lấy đường thấp nhất (closest to bottom = substrate surface)
        # Dùng median của 3 đường thấp nhất để tránh outlier
        cands_sorted = sorted(candidates, reverse=True)
        baseline_y = int(np.median(cands_sorted[:min(3, len(cands_sorted))]))
        return baseline_y

    # Fallback: tìm hàng pixel sáng nhất trong vùng 55%-90%
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scan_y1 = int(h * 0.55)
    scan_y2 = int(h * 0.90)
    row_means = np.mean(gray_full[scan_y1:scan_y2, :], axis=1)
    best_row = int(np.argmax(row_means)) + scan_y1
    return best_row


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TIỀN XỬ LÝ + EDGE DETECTION + CONTOUR
# ═══════════════════════════════════════════════════════════════════════════════
def find_drop_contour(frame: np.ndarray, baseline_y: int):
    """
    Tìm contour giọt trong vùng crop [y_top : baseline_y].

    Trả về: (contour_global, binary_vis, y_top)
      contour_global: tọa độ trong frame gốc
      binary_vis: ảnh binary dùng để debug
    """
    h, w = frame.shape[:2]

    # Crop: bỏ phần text overlay trên (15%) và substrate dưới
    y_top = int(h * 0.15)
    y_bot = baseline_y - 2   # dừng trước baseline 2px
    if y_bot <= y_top + 20:
        return None, np.zeros((10, 10), np.uint8), y_top

    crop_bgr = frame[y_top:y_bot, :]

    # Kênh màu tốt nhất
    gray = best_channel(crop_bgr)

    # CLAHE để tăng tương phản vùng giọt tối
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Blur để giảm noise từ phản quang bên trong giọt
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # ── Threshold: giọt TỐI → THRESH_BINARY_INV ──────────────────────────────
    # Otsu thường không ổn vì histogram bimodal yếu → dùng adaptive hoặc dark-region

    # Cách 1: Otsu inverse (giọt sáng trong binary)
    _, bin_otsu = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Cách 2: Adaptive (tốt hơn khi nền không đều)
    bin_adapt = cv2.adaptiveThreshold(blurred, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 31, 8)

    # Kết hợp: AND → chỉ lấy vùng mà cả 2 cùng nhận là foreground
    binary = cv2.bitwise_and(bin_otsu, bin_adapt)

    # ── Morphological cleanup ─────────────────────────────────────────────────
    # Closing lớn để lấp lỗ phản quang bên trong giọt
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # Opening nhỏ để xóa nhiễu
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    # ── Tìm contour ──────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, binary, y_top

    crop_h, crop_w = crop_bgr.shape[:2]
    min_area = crop_h * crop_w * 0.005   # ít nhất 0.5% vùng crop

    # Lọc theo area VÀ vị trí (contour phải chạm đến gần đáy vùng crop)
    valid = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        _, cy, _, ch = cv2.boundingRect(c)
        bottom_y = cy + ch
        # Contour phải chạm tới ít nhất 75% chiều cao crop
        if bottom_y > crop_h * 0.70:
            valid.append(c)

    if not valid:
        # Fallback: chỉ lọc theo area (bỏ filter vị trí)
        valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        return None, binary, y_top

    # Lấy contour lớn nhất
    drop_c = max(valid, key=cv2.contourArea)

    # Dịch tọa độ về frame gốc
    drop_c_global = drop_c + np.array([[[0, y_top]]])

    return drop_c_global, binary, y_top


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TÍNH GÓC TIẾP XÚC
# ═══════════════════════════════════════════════════════════════════════════════
def compute_contact_angles(drop_contour: np.ndarray,
                            baseline_y: int,
                            fit_height_frac: float = 0.12,
                            poly_deg: int = 2) -> tuple:
    """
    Tính góc tiếp xúc trái và phải dùng polynomial fit cục bộ.

    fit_height_frac: phần trăm chiều cao giọt dùng để fit (mặc định 12%)
                    → tránh fit quá ít điểm (noisy) hoặc quá nhiều (mất cục bộ)
    poly_deg       : bậc polynomial (2 = parabol, đủ cho hầu hết giọt)

    Thuật toán:
    1. Lấy điểm contour gần baseline (y >= baseline_y - fit_height_px)
    2. Chia trái / phải theo x_mid
    3. Fit polynomial x = f(y) bậc poly_deg
    4. Tính derivative dx/dy tại contact point (y = baseline_y)
    5. θ = arctan(|1/(dx/dy)|) = arctan(|dy/dx|)
       Nếu |dx/dy| >> 1 → tiếp tuyến gần thẳng đứng → θ → 90°
    """
    pts = drop_contour.reshape(-1, 2).astype(float)

    # Chiều cao giọt (từ baseline lên đỉnh)
    y_top_drop = pts[:, 1].min()
    drop_height = baseline_y - y_top_drop
    if drop_height < 10:
        return None, None

    fit_height_px = max(20, int(drop_height * fit_height_frac))

    # Chỉ lấy điểm gần baseline
    near_base = pts[pts[:, 1] >= baseline_y - fit_height_px]
    if len(near_base) < 10:
        return None, None

    xs, ys = near_base[:, 0], near_base[:, 1]
    x_min, x_max = xs.min(), xs.max()
    x_mid = (x_min + x_max) / 2

    angles = []
    for side, x_cond in [("left",  xs <= x_mid + 10),
                          ("right", xs >= x_mid - 10)]:
        sx, sy = xs[x_cond], ys[x_cond]
        if len(sx) < 6:
            angles.append(None)
            continue

        # Tìm contact point = điểm có y lớn nhất (chạm baseline)
        idx_contact = np.argmax(sy)
        y_contact = sy[idx_contact]

        # Lấy điểm trong cửa sổ fit phía trên contact point
        win = (sy >= y_contact - fit_height_px) & (sy <= y_contact)
        sx_w, sy_w = sx[win], sy[win]
        if len(sx_w) < 5:
            angles.append(None)
            continue

        # Loại outlier đơn giản (IQR trên x)
        q1, q3 = np.percentile(sx_w, 25), np.percentile(sx_w, 75)
        iqr = q3 - q1 + 1e-6
        mask_iqr = (sx_w >= q1 - 2 * iqr) & (sx_w <= q3 + 2 * iqr)
        sx_w, sy_w = sx_w[mask_iqr], sy_w[mask_iqr]
        if len(sx_w) < 4:
            angles.append(None)
            continue

        try:
            # Fit x = poly(y) — dùng y làm biến độc lập vì contour có thể thẳng đứng
            coeffs = np.polyfit(sy_w, sx_w, poly_deg)
            # Derivative dx/dy tại y = y_contact
            deriv_coeffs = np.polyder(coeffs)
            dxdy = np.polyval(deriv_coeffs, y_contact)

            # Góc giữa tiếp tuyến và baseline (nằm ngang)
            # Slope của tiếp tuyến theo (x, y) = (1, dxdy) ... nhưng trong ảnh y↓
            # θ = arctan(|dy/dx|) = arctan(|1/dxdy|) nếu dxdy≠0
            if abs(dxdy) < 1e-6:
                # Tiếp tuyến thẳng đứng → θ = 90°
                angle_deg = 90.0
            else:
                angle_deg = np.degrees(np.arctan(abs(1.0 / dxdy)))

            # Kiểm tra hướng hợp lý:
            # Bên trái: dxdy > 0 → x tăng khi y tăng → tiếp tuyến nghiêng phải → OK
            # Bên phải: dxdy < 0 → x giảm khi y tăng → tiếp tuyến nghiêng trái → OK
            # Nếu ngược lại → contour sai → reject
            if side == "left" and dxdy < -0.5:
                angles.append(None)
                continue
            if side == "right" and dxdy > 0.5:
                angles.append(None)
                continue

            angle_deg = float(np.clip(angle_deg, 5.0, 175.0))
            angles.append(round(angle_deg, 2))

        except Exception:
            angles.append(None)

    return angles[0], angles[1]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. XỬ LÝ 1 FRAME
# ═══════════════════════════════════════════════════════════════════════════════
def process_frame(img_path: str, debug_dir: str = None) -> dict:
    frame = cv2.imread(img_path)
    if frame is None:
        return {"error": f"Không đọc được ảnh: {img_path}"}

    h, w = frame.shape[:2]
    filename = os.path.basename(img_path)

    # Parse timestamp từ tên file
    ts_match = re.search(r'_t([\d.]+)s', filename)
    timestamp_s = float(ts_match.group(1)) if ts_match else 0.0

    # ── OCR nhiệt độ ──────────────────────────────────────────────────────────
    temperature = ocr_temperature(frame)

    # ── Tìm baseline ──────────────────────────────────────────────────────────
    baseline_y = find_baseline_y(frame)

    # ── Tìm contour giọt ──────────────────────────────────────────────────────
    drop_contour, binary_vis, crop_y_top = find_drop_contour(frame, baseline_y)

    note = ""
    angle_left = angle_right = angle_mean = None

    if drop_contour is None:
        note = "Không tìm thấy giọt"
    else:
        angle_left, angle_right = compute_contact_angles(drop_contour, baseline_y)

        if angle_left is not None and angle_right is not None:
            # Sanity check: 2 góc không được lệch nhau quá 30°
            if abs(angle_left - angle_right) > 35:
                note = f"Góc lệch lớn: L={angle_left} R={angle_right}"
                # Vẫn giữ nhưng đánh dấu
            angle_mean = round((angle_left + angle_right) / 2, 2)
        elif angle_left is not None:
            angle_mean = angle_left
            note = "Chỉ có góc trái"
        elif angle_right is not None:
            angle_mean = angle_right
            note = "Chỉ có góc phải"
        else:
            note = "Không tính được góc"

    result = dict(
        filename=filename,
        timestamp_s=timestamp_s,
        temperature_C=temperature,
        angle_left=angle_left,
        angle_right=angle_right,
        angle_mean=angle_mean,
        baseline_y=baseline_y,
        note=note
    )

    # ── Debug image ────────────────────────────────────────────────────────────
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        dbg = frame.copy()

        # Baseline (đỏ)
        cv2.line(dbg, (0, baseline_y), (w, baseline_y), (0, 0, 255), 2)

        if drop_contour is not None:
            # Contour giọt (xanh lá)
            cv2.drawContours(dbg, [drop_contour], -1, (0, 255, 0), 2)

            # Vẽ contact points
            pts = drop_contour.reshape(-1, 2).astype(float)
            xs, ys = pts[:, 0], pts[:, 1]
            x_mid = (xs.min() + xs.max()) / 2
            left_pts  = pts[xs <= x_mid]
            right_pts = pts[xs >= x_mid]
            if len(left_pts) > 0:
                lcp = left_pts[np.argmax(left_pts[:, 1])]
                cv2.circle(dbg, (int(lcp[0]), int(lcp[1])), 5, (0, 255, 255), -1)
            if len(right_pts) > 0:
                rcp = right_pts[np.argmax(right_pts[:, 1])]
                cv2.circle(dbg, (int(rcp[0]), int(rcp[1])), 5, (255, 0, 255), -1)

        # Binary vis (nhỏ, góc dưới phải)
        bin_rgb = cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR)
        scale_d = 0.22
        bin_small = cv2.resize(bin_rgb, None, fx=scale_d, fy=scale_d)
        bh2, bw2 = bin_small.shape[:2]
        dbg[h - bh2:h, w - bw2:w] = bin_small

        # Label
        temp_str = f"{temperature:.0f}°C" if temperature is not None else "?°C"
        label = (f"L:{angle_left}  R:{angle_right}  "
                 f"Mean:{angle_mean}  T:{temp_str}  base_y:{baseline_y}")
        cv2.putText(dbg, label, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)
        if note:
            cv2.putText(dbg, note, (8, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2)

        cv2.imwrite(os.path.join(debug_dir, "dbg_" + filename), dbg)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NỘI SUY NHIỆT ĐỘ bị thiếu (forward-fill + backward-fill)
# ═══════════════════════════════════════════════════════════════════════════════
def interpolate_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nội suy tuyến tính nhiệt độ từ các frame có OCR thành công.
    Dùng pandas interpolate + ffill + bfill.
    """
    df = df.copy()
    df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce")
    df["temperature_C"] = (
        df["temperature_C"]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 8. XỬ LÝ TOÀN BỘ THƯ MỤC
# ═══════════════════════════════════════════════════════════════════════════════
def process_all_frames(frames_dir: str, output_csv: str,
                       debug: bool = True) -> pd.DataFrame:
    images = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not images:
        print(f"[ERROR] Không có ảnh trong: {frames_dir}")
        return pd.DataFrame()

    print(f"[INFO] Tìm thấy {len(images)} frame, bắt đầu xử lý...")

    out_dir = os.path.dirname(output_csv) or "output"
    debug_dir = os.path.join(out_dir, "debug") if debug else None

    results = []
    for i, img_name in enumerate(images):
        img_path = os.path.join(frames_dir, img_name)
        result = process_frame(img_path, debug_dir=debug_dir)
        results.append(result)
        status = (f"T={result.get('temperature_C')}°C  "
                  f"θ={result.get('angle_mean')}°  "
                  f"base_y={result.get('baseline_y')}  "
                  f"{result.get('note', '')}")
        print(f"  [{i+1:3d}/{len(images)}] {img_name} → {status}")

    df = pd.DataFrame(results)

    # Nội suy nhiệt độ bị thiếu
    n_missing_before = df["temperature_C"].isna().sum()
    df = interpolate_temperature(df)
    n_missing_after  = df["temperature_C"].isna().sum()
    if n_missing_before > 0:
        print(f"[INFO] Nội suy nhiệt độ: {n_missing_before} frame thiếu "
              f"→ còn {n_missing_after} thiếu sau nội suy")

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n[OK] Đã xuất CSV: {output_csv}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    FRAMES_DIR = r"frames"
    OUTPUT_CSV = r"output\wetting_angles.csv"
    DEBUG      = True

    df = process_all_frames(FRAMES_DIR, OUTPUT_CSV, debug=DEBUG)
    print(df.head(10))