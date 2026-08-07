#!/usr/bin/env python3
"""
Bounding Box Annotator for WSI histology images.

Opens OME-TIFF images at pyramid level 3, lets you draw N bounding boxes
with the mouse, and saves them to labels_bbox.csv (x,y = center, w,h = size).

Usage:
    python annotate_bbox.py               # prompts for image name
    python annotate_bbox.py B1-01-A-01    # annotate a specific image (partial match)

Controls:
    Left-click + drag   → draw a bounding box
    z                   → undo last box
    c                   → clear all boxes
    Close window        → confirm & save to CSV
"""

import os
import sys
import csv
import glob
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'export', 'images')
CSV_PATH = os.path.join(PROJECT_ROOT, 'labels_bbox.csv')
PYRAMID_LEVEL = 3  # 0-indexed pyramid level


# ── Lightweight Pyramid TIFF reader ──────────────────────────────────────────

class PyramidTiff:
    """Efficient wrapper for pyramidal OME-TIFF using tifffile."""

    def __init__(self, path, verbose=True):
        self.path = path
        self.verbose = verbose
        self.tif = tifffile.TiffFile(path)
        self._detect_pyramid_structure()
        self._cache_level_info()
        if self.verbose:
            self._print_info()

    # ── detection ────────────────────────────────────────────────────────

    def _detect_pyramid_structure(self):
        self._pyramid_type = None
        self._levels_source = []

        # Option 1: series[0].levels (OME-TIFF with SubIFDs)
        if len(self.tif.series) > 0:
            first = self.tif.series[0]
            if hasattr(first, 'levels') and len(first.levels) > 1:
                self._pyramid_type = 'series_levels'
                self._levels_source = first.levels
                self.n_levels = len(self._levels_source)
                return

        # Option 2: multiple series
        if len(self.tif.series) > 1:
            shapes = [s.shape for s in self.tif.series]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'multiple_series'
                self._levels_source = self.tif.series
                self.n_levels = len(self._levels_source)
                return

        # Option 3: multiple pages
        if len(self.tif.pages) > 1:
            shapes = [p.shape for p in self.tif.pages]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'pages'
                self._levels_source = list(self.tif.pages)
                self.n_levels = len(self._levels_source)
                return

        # Fallback: single level
        self._pyramid_type = 'single'
        if len(self.tif.series) > 0:
            self._levels_source = [self.tif.series[0]]
        else:
            self._levels_source = [self.tif.pages[0]]
        self.n_levels = 1

    @staticmethod
    def _shapes_are_pyramid(shapes):
        if len(shapes) < 2:
            return False
        sizes = []
        for shape in shapes:
            dims = sorted(shape, reverse=True)[:2]
            sizes.append(max(dims))
        for i in range(1, len(sizes)):
            if sizes[i] >= sizes[i - 1]:
                return False
        return True

    # ── caching ───────────────────────────────────────────────────────────

    def _cache_level_info(self):
        self.level_info = []
        base_w, base_h = 0, 0
        for i, level_src in enumerate(self._levels_source):
            h, w, c = self._parse_shape(level_src.shape)
            ds = 1.0 if i == 0 else (base_w / w if w > 0 else 1.0)
            if i == 0:
                base_w, base_h = w, h
            self.level_info.append({
                'index': i,
                'shape': level_src.shape,
                'width': w,
                'height': h,
                'channels': c,
                'downsample': ds,
            })

    @staticmethod
    def _parse_shape(shape):
        if len(shape) == 2:
            return shape[0], shape[1], 1
        if len(shape) == 3:
            if shape[0] <= 4:          # (C, H, W)
                return shape[1], shape[2], shape[0]
            return shape[0], shape[1], shape[2]  # (H, W, C)
        if len(shape) >= 4:            # OME: (T, C, Z, Y, X)
            return shape[-2], shape[-1], shape[1] if shape[1] <= 4 else 1
        return shape[0], 1, 1

    def _print_info(self):
        print(f"  Pyramid type: {self._pyramid_type}, levels: {self.n_levels}")
        for info in self.level_info:
            print(f"    L{info['index']}: {info['width']}×{info['height']} "
                  f"(ds={info['downsample']:.0f}×)")

    # ── reading ───────────────────────────────────────────────────────────

    def read_level(self, level=0):
        level = min(level, self.n_levels - 1)
        info = self.level_info[level]
        if self.verbose:
            print(f"  Reading level {level}: {info['width']}×{info['height']}")
        data = self._levels_source[level].asarray()
        if self.verbose:
            print(f"  Loaded: shape={data.shape}, RAM={data.nbytes/1024/1024:.1f} MB")
        return self._normalize_shape(data)

    def _normalize_shape(self, data):
        data = np.squeeze(data)
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            if data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
                return np.moveaxis(data, 0, -1)   # (C,H,W) → (H,W,C)
            return data
        while data.ndim > 3:
            data = data[0]
        return self._normalize_shape(data)

    def close(self):
        self.tif.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Image normalisation for display ──────────────────────────────────────────

def normalise_for_display(image):
    """Convert any image dtype to uint8 for matplotlib imshow."""
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


# ── Bounding-box annotator ───────────────────────────────────────────────────

class BBoxAnnotator:
    """Matplotlib figure that lets the user draw rectangles over an image."""

    def __init__(self, image, wsi_name, level, downsample):
        self.image = image
        self.wsi_name = wsi_name
        self.level = level
        self.downsample = downsample

        self.img_h, self.img_w = image.shape[:2]

        # Each bbox is stored as (cx, cy, w, h) in relative coords [0, 1]
        self.bboxes = []

        # Per-bbox patches and label artists (for undo)
        self._patches = []      # list of Rectangle
        self._labels = []       # list of Annotation

        # Figure setup
        self.fig, self.ax = plt.subplots(figsize=(14, 11))
        self.fig.canvas.manager.set_window_title(
            f"Annotating: {wsi_name}  [level {level}]"
        )

        h_img, w_img = image.shape[:2]
        if image.ndim == 2:
            self.ax.imshow(image, cmap='gray')
        else:
            self.ax.imshow(image)

        self.ax.set_title(
            f"{wsi_name}  |  Level {level}  |  {w_img}×{h_img} px  "
            f"|  ds={downsample:.0f}×\n"
            f"Draw: left-drag  |  Undo: Z  |  Clear: C  |  "
            f"Close window → save",
            fontsize=9, fontfamily='monospace'
        )

        # Drawing state
        self._start_xy = None

        # ── Pre-create animated temp rectangle (blitting optimisation) ──
        # Instead of creating/destroying on every mouse move, we create it
        # once as an animated artist and only update its geometry.
        self._temp_rect = Rectangle(
            (0, 0), 0, 0,
            linewidth=1.5, edgecolor='lime', facecolor='none',
            linestyle='--', visible=False, animated=True,
        )
        self.ax.add_patch(self._temp_rect)

        # Blitting state
        self._background = None
        self._last_motion_time = 0.0
        self._motion_throttle = 1 / 60.0   # max ~60 fps

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)

        plt.tight_layout()

    # ── blitting helpers ─────────────────────────────────────────────────

    def _on_draw(self, event):
        """Cache the static background after every full redraw."""
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _on_resize(self, event):
        """Force background re-cache on window resize."""
        self._background = None

    def _redraw_full(self):
        """Trigger a full redraw and re-cache the background for blitting."""
        self._background = None
        self._temp_rect.set_visible(False)
        self.fig.canvas.draw_idle()
        # Force immediate draw so background is cached before next motion
        self.fig.canvas.flush_events()

    # ── mouse handlers ────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._start_xy = (event.xdata, event.ydata)
        self._temp_rect.set_visible(False)

    def _on_motion(self, event):
        if self._start_xy is None or event.inaxes != self.ax:
            return

        # ── throttle: skip frames that arrive too fast ────────────────
        now = time.perf_counter()
        if now - self._last_motion_time < self._motion_throttle:
            return
        self._last_motion_time = now

        # If background isn't cached yet, do a full draw first
        if self._background is None:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        x0, y0 = self._start_xy
        w = event.xdata - x0
        h = event.ydata - y0

        # ── blit: restore background, draw only the animated rectangle ──
        self.fig.canvas.restore_region(self._background)

        self._temp_rect.set_xy((x0, y0))
        self._temp_rect.set_width(w)
        self._temp_rect.set_height(h)
        self._temp_rect.set_visible(True)
        self.ax.draw_artist(self._temp_rect)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def _on_release(self, event):
        if self._start_xy is None or event.inaxes != self.ax:
            self._start_xy = None
            return

        x0, y0 = self._start_xy
        x1, y1 = event.xdata, event.ydata
        self._start_xy = None

        # Hide temp rect (it will stay hidden in the background cache)
        self._temp_rect.set_visible(False)

        # Compute canonical box (min_x, min_y, w, h)
        min_x = min(x0, x1)
        min_y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)

        # Reject tiny rectangles (accidental clicks)
        if w < 5 and h < 5:
            self._redraw_full()
            return

        cx = min_x + w / 2.0
        cy = min_y + h / 2.0

        # Normalise to [0, 1] relative to image dimensions
        rcx = cx / self.img_w
        rcy = cy / self.img_h
        rw = w / self.img_w
        rh = h / self.img_h

        idx = len(self.bboxes) + 1
        core_label = f'core_{idx}'
        self.bboxes.append((rcx, rcy, rw, rh))

        # Permanent red rectangle (drawn in pixel space)
        rect = Rectangle(
            (min_x, min_y), w, h,
            linewidth=2.0, edgecolor='red', facecolor='none',
        )
        self.ax.add_patch(rect)
        self._patches.append(rect)

        # Label at center
        lbl = self.ax.annotate(
            core_label,
            xy=(cx, cy),
            color='red', fontsize=7, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8),
        )
        self._labels.append(lbl)

        # Full redraw to bake the new rectangle into the background cache
        self._redraw_full()
        print(f"  ✓ {core_label}: center=({cx:.1f}, {cy:.1f}) px  "
              f"({rcx:.4f}, {rcy:.4f}) rel  |  "
              f"size={w:.1f}×{h:.1f} px  ({rw:.4f}×{rh:.4f}) rel")

    # ── keyboard handler ──────────────────────────────────────────────────

    def _on_key(self, event):
        if event.key == 'z':
            self._undo_last()
        elif event.key == 'c':
            self._clear_all()

    def _undo_last(self):
        if not self.bboxes:
            return
        removed = self.bboxes.pop()
        self._patches.pop().remove()
        self._labels.pop().remove()
        self._redraw_full()
        print(f"  ✗ Undo core_{len(self.bboxes) + 1}: "
              f"center=({removed[0]:.1f}, {removed[1]:.1f})")

    def _clear_all(self):
        n = len(self.bboxes)
        self.bboxes.clear()
        for p in self._patches:
            p.remove()
        self._patches.clear()
        for lbl in self._labels:
            lbl.remove()
        self._labels.clear()
        self._redraw_full()
        print(f"  ✗ Cleared {n} bbox(es)")

    # ── run ───────────────────────────────────────────────────────────────

    def run(self):
        """Block until the user closes the figure window."""
        plt.show(block=True)

    def get_bboxes(self):
        """Return list of (cx, cy, w, h) in relative [0, 1] coordinates."""
        return list(self.bboxes)


# ── CSV persistence ──────────────────────────────────────────────────────────

CSV_FIELDS = ['WSI_NAME', 'core_id', 'x', 'y', 'w', 'h']


def load_completed(csv_path):
    """Return set of WSI names that already have entries in the CSV."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('WSI_NAME', '').strip()
            if name:
                completed.add(name)
    return completed


def save_bboxes(csv_path, wsi_name, bboxes):
    """
    Write or update entries for *wsi_name* in the CSV.
    Existing rows for other WSIs are preserved.
    Previously saved rows for *wsi_name* are **replaced**.
    """
    # Read everything except rows belonging to wsi_name
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('WSI_NAME', '').strip() != wsi_name:
                    existing.append(row)

    # Append current bboxes (relative coords [0, 1] → 6 decimal places)
    for i, (cx, cy, w, h) in enumerate(bboxes, start=1):
        existing.append({
            'WSI_NAME': wsi_name,
            'core_id': f'core_{i}',
            'x': f'{cx:.6f}',
            'y': f'{cy:.6f}',
            'w': f'{w:.6f}',
            'h': f'{h:.6f}',
        })

    # Write out
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing)

    print(f"  💾 Saved {len(bboxes)} bbox(es) for {wsi_name} → {csv_path}")


# ── File discovery ───────────────────────────────────────────────────────────

def find_image_files(directory):
    """Return sorted list of *.tif / *.tiff paths in *directory*."""
    patterns = ('*.tif', '*.tiff')
    files = set()
    for pat in patterns:
        files.update(glob.glob(os.path.join(directory, pat)))
    return sorted(files)


def extract_wsi_name(filepath):
    """'B1-01-A-01-001.ome.tif' → 'B1-01-A-01-001'."""
    name = os.path.basename(filepath)
    name = os.path.splitext(name)[0]       # drop .tif
    if name.endswith('.ome'):
        name = name[:-4]                    # drop .ome
    return name


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(IMAGES_DIR):
        print(f"ERROR: images directory not found → {IMAGES_DIR}")
        sys.exit(1)

    all_files = find_image_files(IMAGES_DIR)
    if not all_files:
        print(f"ERROR: no .tif / .tiff files in {IMAGES_DIR}")
        sys.exit(1)

    print(f"📁 Images dir : {IMAGES_DIR}")
    print(f"📄 CSV output : {CSV_PATH}")
    print(f"🖼️  Found       : {len(all_files)} image(s)")

    completed = load_completed(CSV_PATH)
    if completed:
        print(f"✅ Already done : {len(completed)} image(s)")

    # ── decide which images to annotate ────────────────────────────────────
    if len(sys.argv) > 1:
        wsi_filter = sys.argv[1]
        print(f"🔍 Filter      : \"{wsi_filter}\" (from command line)")
        # When a specific image is requested, re-annotate even if already done
        target_files = [f for f in all_files
                        if wsi_filter in os.path.basename(f)]
        if not target_files:
            print(f"No files matching \"{wsi_filter}\"")
            sys.exit(1)
    else:
        # Show list (✅ markers are purely informational)
        print()
        for i, fp in enumerate(all_files):
            name = extract_wsi_name(fp)
            tag = " ✅" if name in completed else ""
            print(f"  [{i:3d}]  {name}{tag}")

        raw = input(
            "\nEnter WSI index or name (press Enter = all unannotated): "
        ).strip()

        if raw == "":
            # ── Enter → annotate only the unannotated ones ────────────
            target_files = [f for f in all_files
                            if extract_wsi_name(f) not in completed]
        else:
            # ── explicit index or name → honour it, annotated or not ─
            try:
                idx = int(raw)
                if 0 <= idx < len(all_files):
                    target_files = [all_files[idx]]
                else:
                    print(f"Index {idx} out of range (0–{len(all_files) - 1})")
                    sys.exit(1)
            except ValueError:
                # Substring match against filename
                target_files = [f for f in all_files
                                if raw in os.path.basename(f)]
                if not target_files:
                    print(f"No files matching \"{raw}\"")
                    sys.exit(1)

    print(f"\n🎯 To annotate : {len(target_files)} image(s)\n")

    # ── annotate each image ──────────────────────────────────────────────
    for fp in target_files:
        wsi_name = extract_wsi_name(fp)
        print(f"{'─' * 60}")
        print(f"📷 {os.path.basename(fp)}")
        print(f"   WSI name: {wsi_name}")

        try:
            with PyramidTiff(fp, verbose=True) as ptiff:
                actual_level = min(PYRAMID_LEVEL, ptiff.n_levels - 1)
                if actual_level != PYRAMID_LEVEL:
                    print(f"   ⚠ Level {PYRAMID_LEVEL} not available; "
                          f"using level {actual_level}")

                image = ptiff.read_level(actual_level)
                ds = ptiff.level_info[actual_level]['downsample']

                image_disp = normalise_for_display(image)

                annotator = BBoxAnnotator(image_disp, wsi_name,
                                          actual_level, ds)
                annotator.run()          # blocks until window is closed
                bboxes = annotator.get_bboxes()

                if bboxes:
                    save_bboxes(CSV_PATH, wsi_name, bboxes)
                    completed.add(wsi_name)
                else:
                    print(f"   ⚠ No bboxes drawn — skipping")

            plt.close('all')

        except Exception as exc:
            print(f"   ❌ ERROR: {exc}")
            import traceback
            traceback.print_exc()
            plt.close('all')
            continue

    print(f"\n{'═' * 60}")
    print(f"✔  Done.  Labels → {CSV_PATH}")


if __name__ == '__main__':
    main()
