"""
run_all.py
----------
Chạy toàn bộ pipeline một lệnh:
  1. Tách frame từ video
  2. Đo góc ướt + OCR nhiệt độ
  3. Xuất CSV
  4. Vẽ biểu đồ
"""

import os
from extract_frames import extract_frames
from measure_angle  import process_all_frames
from plot_results   import plot_wetting_angle

# ══════════════════════════════════════════════
# CẤU HÌNH — chỉnh tại đây
# ══════════════════════════════════════════════
VIDEO_PATH = r"videos\20250801-Anh-30Sn_SideView_1.mpg"
FRAMES_DIR    = r"frames"
OUTPUT_DIR    = r"output"
INTERVAL_SEC  = 5.0        # lấy 1 frame mỗi 1 giây
DEBUG_IMAGES  = True       # lưu ảnh debug để kiểm tra kết quả
SAMPLE_NAME   = "30Sn-Snln-SideView"
# ══════════════════════════════════════════════

CSV_PATH = os.path.join(OUTPUT_DIR, "wetting_angles.csv")

print("=" * 55)
print("  WETTING ANGLE ANALYSIS PIPELINE")
print("=" * 55)

print("\n[STEP 1] Tách frame từ video...")
n_frames = extract_frames(VIDEO_PATH, FRAMES_DIR, INTERVAL_SEC)

print(f"\n[STEP 2] Đo góc ướt từ {n_frames} frame...")
df = process_all_frames(FRAMES_DIR, CSV_PATH, debug=DEBUG_IMAGES)

print(f"\n[STEP 3] Vẽ biểu đồ...")
plot_wetting_angle(CSV_PATH, OUTPUT_DIR, SAMPLE_NAME)

print("\n" + "=" * 55)
print("  HOÀN THÀNH!")
print(f"  CSV  → {CSV_PATH}")
print(f"  Plot → {OUTPUT_DIR}\\{SAMPLE_NAME}_contact_angle.png")
print("=" * 55)
