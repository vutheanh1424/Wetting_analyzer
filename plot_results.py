"""
plot_results.py
---------------
Đọc CSV kết quả và vẽ biểu đồ góc ướt vs nhiệt độ
cho paper/báo cáo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys


def plot_wetting_angle(csv_path: str, output_dir: str, sample_name: str = ""):
    """
    Vẽ biểu đồ contact angle vs nhiệt độ từ CSV.
    """
    df = pd.read_csv(csv_path)

    # Lọc hàng có đủ dữ liệu
    df_clean = df.dropna(subset=["temperature_C", "angle_mean"]).copy()
    df_clean = df_clean[df_clean["angle_mean"] > 0]
    df_clean = df_clean.sort_values("temperature_C")

    if df_clean.empty:
        print("[ERROR] Không có dữ liệu hợp lệ để vẽ biểu đồ.")
        return

    temp   = df_clean["temperature_C"].values
    angle  = df_clean["angle_mean"].values
    a_left = df_clean["angle_left"].values
    a_right= df_clean["angle_right"].values

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Wetting Angle vs Temperature  —  {sample_name}", fontsize=13, y=1.01)

    # Plot 1: Mean contact angle vs Temperature
    ax1 = axes[0]
    ax1.plot(temp, angle, 'o-', color='#2563EB', linewidth=2,
             markersize=6, label='Mean contact angle', zorder=3)

    # Shaded error band (left/right)
    if a_left is not None and a_right is not None:
        try:
            al = a_left.astype(float)
            ar = a_right.astype(float)
            valid = ~(np.isnan(al) | np.isnan(ar))
            ax1.fill_between(temp[valid], al[valid], ar[valid],
                             alpha=0.15, color='#2563EB', label='Left–Right range')
        except Exception:
            pass

    ax1.set_xlabel("Temperature (°C)", fontsize=11)
    ax1.set_ylabel("Contact Angle (°)", fontsize=11)
    ax1.set_title("Contact Angle vs Temperature", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Plot 2: Angle vs Time (giữ nguyên thứ tự thời gian)
    df_time = df.dropna(subset=["timestamp_s", "angle_mean"]).copy()
    df_time = df_time[df_time["angle_mean"] > 0].sort_values("timestamp_s")

    ax2 = axes[1]
    sc = ax2.scatter(df_time["timestamp_s"] / 60, df_time["angle_mean"],
                     c=df_time["temperature_C"], cmap='plasma',
                     s=40, zorder=3, label='Contact angle')
    ax2.plot(df_time["timestamp_s"] / 60, df_time["angle_mean"],
             '-', color='gray', linewidth=1, alpha=0.5)

    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Temperature (°C)", fontsize=9)
    ax2.set_xlabel("Time (min)", fontsize=11)
    ax2.set_ylabel("Contact Angle (°)", fontsize=11)
    ax2.set_title("Contact Angle vs Time (color = temperature)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Lưu file
    os.makedirs(output_dir, exist_ok=True)
    fname = sample_name.replace(" ", "_") if sample_name else "wetting_result"
    out_png = os.path.join(output_dir, f"{fname}_contact_angle.png")
    out_svg = os.path.join(output_dir, f"{fname}_contact_angle.svg")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    print(f"[OK] Đã lưu biểu đồ:\n  {out_png}\n  {out_svg}")
    plt.show()


if __name__ == "__main__":
    # ── Chỉnh 3 dòng này ──────────────────────────────────────────────
    CSV_PATH    = r"output\wetting_angles.csv"
    OUTPUT_DIR  = r"output"
    SAMPLE_NAME = "30Sn-Snln"   # tên hiện trên biểu đồ
    # ──────────────────────────────────────────────────────────────────

    plot_wetting_angle(CSV_PATH, OUTPUT_DIR, SAMPLE_NAME)
