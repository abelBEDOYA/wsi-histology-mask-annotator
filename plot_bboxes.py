#!/usr/bin/env python3
"""
Simple script: for every .tif in dataset/export/images/,
open it at pyramid level 4, draw the bounding boxes from
labels_bbox.csv and core_mapping.csv, and save the result as
{wsi_name}_cores_bbox.png.

Usage:
    python plot_bboxes.py
"""

import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # headless – no window
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile

# ── Paths (same convention as annotate_bbox.py) ──────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR   = os.path.join(PROJECT_ROOT, "dataset", "export", "images")
CSV_PATHS = [
    os.path.join(PROJECT_ROOT, "labels_bbox.csv"),
    os.path.join(PROJECT_ROOT, "core_mapping.csv"),
    os.path.join(PROJECT_ROOT, "dataset", "export", "core_mapping.csv"),
]
OUT_DIR      = PROJECT_ROOT                        # save plots next to the script
PYRAMID_LEVEL = 4                                  # more reduced (higher downsample)

# ── Lightweight PyramidTiff (copy of the one in annotate_bbox.py) ────────

class PyramidTiff:
    """Minimal pyramidal OME-TIFF reader."""

    def __init__(self, path):
        self.tif = tifffile.TiffFile(path)
        self._detect_pyramid_structure()
        self._cache_level_info()

    def _detect_pyramid_structure(self):
        # Option 1: series[0].levels
        if len(self.tif.series) > 0 and hasattr(self.tif.series[0], "levels"):
            if len(self.tif.series[0].levels) > 1:
                self._src = self.tif.series[0].levels
                self.n_levels = len(self._src)
                return
        # Option 2: multiple series
        if len(self.tif.series) > 1:
            shapes = [s.shape for s in self.tif.series]
            if self._shapes_are_pyramid(shapes):
                self._src = self.tif.series
                self.n_levels = len(self._src)
                return
        # Fallback: single series
        self._src = [self.tif.series[0]] if len(self.tif.series) > 0 else [self.tif.pages[0]]
        self.n_levels = 1

    @staticmethod
    def _shapes_are_pyramid(shapes):
        sizes = [max(sorted(s, reverse=True)[:2]) for s in shapes]
        return len(sizes) >= 2 and all(sizes[i] < sizes[i - 1] for i in range(1, len(sizes)))

    def _cache_level_info(self):
        self.level_info = []
        base_w = base_h = 0
        for i, src in enumerate(self._src):
            h, w, c = self._parse_shape(src.shape)
            ds = 1.0 if i == 0 else (base_w / w if w > 0 else 1.0)
            if i == 0:
                base_w, base_h = w, h
            self.level_info.append(
                {"index": i, "width": w, "height": h, "channels": c, "downsample": ds}
            )

    @staticmethod
    def _parse_shape(shape):
        if len(shape) == 2:
            return shape[0], shape[1], 1
        if len(shape) == 3:
            if shape[0] <= 4:          # (C, H, W)
                return shape[1], shape[2], shape[0]
            return shape[0], shape[1], shape[2]
        if len(shape) >= 4:            # (T, C, Z, Y, X)
            return shape[-2], shape[-1], shape[1] if shape[1] <= 4 else 1
        return shape[0], 1, 1

    def read_level(self, level=0):
        level = min(level, self.n_levels - 1)
        info = self.level_info[level]
        print(f"  → level {level}: {info['width']}×{info['height']}")
        data = self._src[level].asarray()
        return self._normalise(data)

    def _normalise(self, data):
        data = np.squeeze(data)
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            if data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
                return np.moveaxis(data, 0, -1)   # (C,H,W) → (H,W,C)
            return data                             # (H,W,C) — already fine
        while data.ndim > 3:
            data = data[0]
        return self._normalise(data)

    def close(self):
        self.tif.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# ── Helpers ──────────────────────────────────────────────────────────────

def normalise_for_display(image):
    """uint16/float32 → uint8 for matplotlib."""
    if image.dtype == np.uint8:
        return image.copy()
    img = image.astype(np.float32)
    if image.dtype == np.uint16:
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        img = (img - vmin) / max(vmax - vmin, 1.0)
    else:
        img = (img - img.min()) / max(img.max() - img.min(), 1e-8)
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def extract_wsi_name(filepath):
    """'B1-01-A-01-001.ome.tif' → 'B1-01-A-01-001'."""
    name = os.path.basename(filepath)
    name = os.path.splitext(name)[0]
    if name.endswith(".ome"):
        name = name[:-4]
    return name

def load_bboxes_for_wsi(wsi_name):
    """Return list of (core_id, cx, cy, w, h) in relative coords [0,1].

    Reads from all configured CSV_PATHS and merges results.
    Deduplicates by core_id within the same WSI.
    """
    seen = set()
    entries = []
    for csv_path in CSV_PATHS:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("anonymous_code", "").strip() == wsi_name:
                    cid = row["core_id"]
                    if cid in seen:
                        continue
                    seen.add(cid)
                    entries.append((
                        cid,
                        float(row["x"]),
                        float(row["y"]),
                        float(row["w"]),
                        float(row["h"]),
                    ))
    return entries

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    tif_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.tif")) +
                       glob.glob(os.path.join(IMAGES_DIR, "*.tiff")))
    if not tif_files:
        print(f"❌ No .tif files found in {IMAGES_DIR}")
        return

    print(f"📁 Found {len(tif_files)} TIFF(s)\n")

    for fp in tif_files:
        wsi_name = extract_wsi_name(fp)
        bboxes = load_bboxes_for_wsi(wsi_name)

        if not bboxes:
            print(f"⏭  {wsi_name} — no bboxes in CSV, skipping")
            continue

        print(f"{'─' * 55}")
        print(f"📷 {os.path.basename(fp)}  ({len(bboxes)} bbox(es))")

        # Open at level 3
        with PyramidTiff(fp) as ptiff:
            actual_level = min(PYRAMID_LEVEL, ptiff.n_levels - 1)
            image = ptiff.read_level(actual_level)
            ds = ptiff.level_info[actual_level]["downsample"]

        image_disp = normalise_for_display(image)
        h_img, w_img = image_disp.shape[:2]

        # ── Plot ─────────────────────────────────────────────────────
        dpi = 150
        fig_w = w_img / dpi
        fig_h = h_img / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        if image_disp.ndim == 2:
            ax.imshow(image_disp, cmap="gray")
        else:
            ax.imshow(image_disp)

        ax.set_title(f"{wsi_name}  |  Level {actual_level}  |  "
                     f"{w_img}×{h_img} px  |  ds={ds:.0f}×  |  "
                     f"{len(bboxes)} cores",
                     fontsize=14, fontweight="bold", fontfamily="monospace")

        # Draw every bounding box (relative → pixel coords)
        for core_id, rcx, rcy, rw, rh in bboxes:
            cx_px = rcx * w_img
            cy_px = rcy * h_img
            bw_px = rw  * w_img
            bh_px = rh  * h_img

            x0 = cx_px - bw_px / 2.0
            y0 = cy_px - bh_px / 2.0

            rect = Rectangle(
                (x0, y0), bw_px, bh_px,
                linewidth=5.0, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)

            ax.annotate(
                core_id,
                xy=(cx_px, cy_px),
                color="red", fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", alpha=0.8),
            )

        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save
        out_path = os.path.join(OUT_DIR, f"{wsi_name}_cores_bbox.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  💾 Saved → {out_path}")

    print(f"\n{'═' * 55}")
    print("✔  Done.")

if __name__ == "__main__":
    main()
