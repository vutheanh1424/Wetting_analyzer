"""
extract_frames.py
-----------------
Tách frame từ video Side View theo interval (mặc định 1 frame/5 giây)
"""

import cv2
import os
import sys

def extract_frames(video_path: str, output_dir: str, interval_sec: float = 5.0):
    """
    Tách frame từ video.
    
    Args:
        video_path  : đường dẫn tới file video (.mp4, .avi, ...)
        output_dir  : thư mục lưu frame ảnh PNG
        interval_sec: lấy 1 frame mỗi bao nhiêu giây (mặc định 5s)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video: {video_path}")
        sys.exit(1)

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_frames / fps
    interval_frames = int(fps * interval_sec)

    print(f"[INFO] Video: {os.path.basename(video_path)}")
    print(f"[INFO] FPS={fps:.1f}  |  Tổng frames={total_frames}  |  Thời lượng={duration/60:.1f} phút")
    print(f"[INFO] Lấy 1 frame mỗi {interval_sec}s → interval={interval_frames} frames")

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            timestamp_sec = frame_idx / fps
            filename = f"frame_{saved:04d}_t{timestamp_sec:.1f}s.png"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[OK] Đã lưu {saved} frame vào: {output_dir}")
    return saved


if __name__ == "__main__":
    # ── Chỉnh 2 dòng này ──────────────────────────────────────────────
    VIDEO_PATH   = r"videos\20250801-Anh-30Sn_SideView_1.mpg"
    FRAMES_DIR   = r"frames"
    INTERVAL_SEC = 5.0   # đổi thành 2.0 nếu muốn dày hơn
    # ──────────────────────────────────────────────────────────────────

    extract_frames(VIDEO_PATH, FRAMES_DIR, INTERVAL_SEC)
