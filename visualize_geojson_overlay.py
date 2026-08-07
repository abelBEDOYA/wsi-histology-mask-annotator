#!/usr/bin/env python3
"""
Visualize GeoJSON annotations overlaid on pyramidal OME-TIFF images.

Loads a TIFF image at a specified pyramid level and overlays the corresponding
GeoJSON annotations (exported by export_geojson_cropped.groovy) with transparent
fills and colored outlines per class.

Coordinates in the GeoJSON are in relative [0,1] space (independent of WSI
resolution), and are mapped to display pixels by multiplying by the rendered
image dimensions at the chosen pyramid level.

Usage:
    # Interactive: list images, choose one
    python visualize_geojson_overlay.py /path/to/dataset

    # Specific image and level
    python visualize_geojson_overlay.py /path/to/dataset --name B1-01-A-01-001 --level 2

    # Auto-level (fits in ~4MP), specific image
    python visualize_geojson_overlay.py /path/to/dataset --name B1-01-A-01-001

    # Save to file instead of displaying
    python visualize_geojson_overlay.py /path/to/dataset --name B1-01-A-01-001 --save overlay.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba

# Local import: PyramidTiff from the same directory
from qupath_handler import PyramidTiff


# =============================================================================
# Class color map (matching CLASS_COLORS_HEX from qupath_handler.py)
# =============================================================================

CLASS_COLORS_HEX: List[str] = [
    "#000000",  # 0: Background (black)
    "#B83B5E",  # 1: Tumor
    "#F38181",  # 2: Bening gland
    "#AA96DA",  # 3: Blood vessels
    "#FCBAD3",  # 4: Fibromuscular bundles
    "#FF6B6B",  # 5: Abnormal secretions
    "#9B59B6",  # 6: Contamination with another tissue
    "#FAE3D9",  # 7: Prominent nucleolus
    "#FF9F43",  # 8: Immune cells
    "#E9D5CA",  # 9: Nerve
    "#8B4513",  # 10: Artifact
    "#FFD93D",  # 11: Seminal vesicle
    "#6BCB77",  # 12: Adipose tissue
    "#4D96FF",  # 13: Normal secretions
    "#C9B1FF",  # 14: Stromal retraction spaces
    "#FF6B6B",  # 15: Muscle
    "#A66CFF",  # 16: Foreign body contamination
    "#FFB830",  # 17: HGPIN
    "#00B4D8",  # 18: Calcifications
    "#E76F51",  # 19: Intestinal glands and mucus
    "#F4A261",  # 20: PNI
    "#E63946",  # 21: Hemorrahage
    "#2A9D8F",  # 22: Intraductal carcinoma
    "#264653",  # 23: Necrosis
    "#8AB17D",  # 24: Mitosis
    "#B5838D",  # 25: Nerve ganglion
    "#FFB4A2",  # 26: Atypical intraductal proliferation
    "#B56576",  # 27: Red blood cells
    "#6D597A",  # 28: Stroma
]

CLASS_NAMES: Dict[int, str] = {
    0: "Background",
    1: "Tumor",
    2: "Bening gland",
    3: "Blood vessels",
    4: "Fibromuscular bundles",
    5: "Abnormal secretions",
    6: "Contamination with another tissue",
    7: "Prominent nucleolus",
    8: "Immune cells",
    9: "Nerve",
    10: "Artifact",
    11: "Seminal vesicle",
    12: "Adipose tissue",
    13: "Normal secretions",
    14: "Stromal retraction spaces",
    15: "Muscle",
    16: "Foreign body contamination",
    17: "High grade prostatic intraepithelial neoplasia (HGPIN)",
    18: "Calcifications",
    19: "Intestinal glands and mucus",
    20: "Perineural invasion (PNI)",
    21: "Hemorrahage",
    22: "Intraductal carcinoma",
    23: "Necrosis",
    24: "Mitosis",
    25: "Nerve ganglion",
    26: "Atypical intraductal proliferation",
    27: "Red blood cells",
    28: "Stroma",
}


# =============================================================================
# GeoJSON loading
# =============================================================================

def load_geojson(path: Path) -> dict:
    """Load a GeoJSON file and return parsed dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def geojson_features_to_patches(
    geojson: dict,
    display_width: int,
    display_height: int,
    visible_class_ids: Optional[set] = None,
) -> Tuple[List[MPLPolygon], List[dict], List[Tuple[float, float, float, float]]]:
    """
    Convert GeoJSON features to matplotlib Polygon patches + per-patch metadata.

    Coordinates in the GeoJSON are relative [0,1] and are multiplied by
    the display dimensions to map to pixel positions on the rendered image.

    Args:
        geojson: Parsed GeoJSON FeatureCollection.
        display_width: Width of the displayed image in pixels.
        display_height: Height of the displayed image in pixels.
        visible_class_ids: Set of class IDs to include (None = all).

    Returns:
        (patches, props_list, rgba_colors) tuple
    """
    patches: List[MPLPolygon] = []
    props_list: List[dict] = []
    rgba_colors: List[Tuple[float, float, float, float]] = []

    for feature in geojson.get("features", []):
        geometry = feature.get("geometry")
        if geometry is None:
            continue

        props = feature.get("properties", {})
        class_id = props.get("classification", {}).get("id", 0)

        if visible_class_ids is not None and class_id not in visible_class_ids:
            continue

        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")

        if coords is None:
            continue

        # Convert relative coordinates [0,1] to display pixel space
        def scale_ring(ring):
            """Scale a single ring from relative [0,1] to display pixels."""
            return [(pt[0] * display_width, pt[1] * display_height) for pt in ring]

        try:
            if geom_type == "Polygon":
                # coordinates: [exterior_ring, ...interior_rings]
                exterior = scale_ring(coords[0])
                interiors = [scale_ring(r) for r in coords[1:]]
                for ring in [exterior] + interiors:
                    if len(ring) < 3:
                        break
                else:
                    patch = MPLPolygon(exterior, closed=True)
                    patches.append(patch)
                    props_list.append(props)
                    color_hex = CLASS_COLORS_HEX[class_id] if 0 <= class_id < len(CLASS_COLORS_HEX) else "#CCCCCC"
                    rgba_colors.append(to_rgba(color_hex))

            elif geom_type == "MultiPolygon":
                # coordinates: [[exterior_ring, ...], [exterior_ring, ...], ...]
                for polygon_coords in coords:
                    exterior = scale_ring(polygon_coords[0])
                    if len(exterior) < 3:
                        continue
                    interiors = [scale_ring(r) for r in polygon_coords[1:]]
                    for ring in [exterior] + interiors:
                        if len(ring) < 3:
                            break
                    else:
                        patch = MPLPolygon(exterior, closed=True)
                        patches.append(patch)
                        props_list.append(props)
                        color_hex = CLASS_COLORS_HEX[class_id] if 0 <= class_id < len(CLASS_COLORS_HEX) else "#CCCCCC"
                        rgba_colors.append(to_rgba(color_hex))

            elif geom_type == "GeometryCollection":
                # Recursively handle nested geometries
                for sub_geom in geometry.get("geometries", []):
                    # Wrap as a single-feature GeoJSON and recurse
                    sub_feature = {
                        "type": "Feature",
                        "geometry": sub_geom,
                        "properties": props,
                    }
                    sub_geojson = {"type": "FeatureCollection", "features": [sub_feature]}
                    sub_patches, sub_props, sub_colors = geojson_features_to_patches(
                        sub_geojson, display_width, display_height, visible_class_ids
                    )
                    patches.extend(sub_patches)
                    props_list.extend(sub_props)
                    rgba_colors.extend(sub_colors)

        except (IndexError, ValueError) as e:
            print(f"  [WARN] Skipping malformed geometry in feature: {e}")
            continue

    return patches, props_list, rgba_colors


# =============================================================================
# Visualization
# =============================================================================

def visualize(
    dataset_dir: Path,
    image_name: Optional[str] = None,
    level: Optional[int] = None,
    alpha: float = 0.25,
    linewidth: float = 1.0,
    geojson_subdir: str = "vector_labels",
    images_subdir: str = "images",
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """
    Main visualization: load TIFF + GeoJSON, overlay, and display/save.

    Args:
        dataset_dir: Root export directory (contains images/, geojson/).
        image_name: Specific image base name, or None for interactive list.
        level: Pyramid level (None = auto, fits in ~4MP).
        alpha: Fill transparency for polygons.
        linewidth: Outline width for polygons.
        geojson_subdir: Subdirectory containing .geojson files.
        images_subdir: Subdirectory containing .ome.tif files.
        save_path: If set, save figure to this path instead of displaying.
        dpi: DPI for saved figures.
    """
    images_dir = dataset_dir / images_subdir
    geojson_dir = dataset_dir / geojson_subdir

    if not images_dir.is_dir():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    if not geojson_dir.is_dir():
        print(f"ERROR: GeoJSON directory not found: {geojson_dir}")
        sys.exit(1)

    # --- Resolve image name ---
    if image_name is None:
        # List available pairs (image + geojson both exist)
        tif_files = sorted(images_dir.glob("*.ome.tif"))
        available = []
        for tif in tif_files:
            base = tif.stem  # e.g. "B1-01-A-01-001" from "B1-01-A-01-001.ome.tif"
            # Also try without .ome: "B1-01-A-01-001.ome" -> "B1-01-A-01-001"
            if base.endswith(".ome"):
                base = base[:-4]
            geojson_path = geojson_dir / f"{base}_annotations.geojson"
            if geojson_path.exists():
                available.append(base)

        if not available:
            print("No image/geojson pairs found.")
            print(f"  Images dir: {images_dir}")
            print(f"  GeoJSON dir: {geojson_dir}")
            sys.exit(1)

        print("Available image/geojson pairs:")
        for i, name in enumerate(available):
            print(f"  [{i}] {name}")

        try:
            choice = input("\nSelect image number (or name): ").strip()
            if choice.isdigit():
                idx = int(choice)
                image_name = available[idx]
            else:
                image_name = choice
        except (ValueError, IndexError, KeyboardInterrupt):
            print("Invalid selection.")
            sys.exit(1)

    # --- Find files ---
    tif_path = None
    for candidate in [
        images_dir / f"{image_name}.ome.tif",
        images_dir / f"{image_name}.ome.tiff",
        images_dir / f"{image_name}.tif",
        images_dir / f"{image_name}.tiff",
    ]:
        if candidate.exists():
            tif_path = candidate
            break

    geojson_path = None
    for candidate in [
        geojson_dir / f"{image_name}_annotations.geojson",
        geojson_dir / f"{image_name}.geojson",
    ]:
        if candidate.exists():
            geojson_path = candidate
            break

    if tif_path is None:
        print(f"ERROR: TIFF not found for '{image_name}' in {images_dir}")
        sys.exit(1)
    if geojson_path is None:
        print(f"ERROR: GeoJSON not found for '{image_name}' in {geojson_dir}")
        sys.exit(1)

    print(f"\nImage:   {tif_path}")
    print(f"GeoJSON: {geojson_path}")

    # --- Load TIFF ---
    print("\nLoading TIFF...")
    tiff = PyramidTiff(str(tif_path), verbose=True)

    # Determine level
    if level is None:
        level = tiff.get_level_for_display(max_pixels=4_000_000)
    level = min(level, tiff.n_levels - 1)

    downsample = tiff.level_info[level]["downsample"]
    img = tiff.read_level(level)

    # RGB normalization for display
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img_display = img[:, :, :3].astype(np.float32)
        # Normalize to [0, 1] if needed
        if img_display.max() > 1.0:
            img_display /= 255.0
    elif img.ndim == 2:
        img_display = img.astype(np.float32)
        if img_display.max() > 1.0:
            img_display /= 255.0
        img_display = np.stack([img_display] * 3, axis=-1)
    else:
        img_display = img.astype(np.float32)
        if img_display.max() > 1.0:
            img_display /= 255.0

    h, w = img_display.shape[:2]
    print(f"Display: {w} x {h} (level {level}, downsample {downsample:.0f}x)")

    tiff.close()

    # --- Load GeoJSON ---
    print("Loading GeoJSON...")
    geojson = load_geojson(geojson_path)
    n_features = len(geojson.get("features", []))
    print(f"  Features: {n_features}")

    # --- Convert to patches ---
    print("Converting geometries to overlay...")
    patches, props_list, colors = geojson_features_to_patches(geojson, w, h)
    print(f"  Patches created: {len(patches)}")

    if not patches:
        print("WARNING: No valid polygons found in GeoJSON.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img_display, origin="upper", interpolation="bilinear")

    if patches:
        collection = PatchCollection(
            patches,
            facecolors=colors,
            edgecolors=colors,
            linewidths=linewidth,
            alpha=alpha,
        )
        ax.add_collection(collection)

    # --- Legend ---
    # Collect unique classes present
    seen_classes: Dict[int, str] = {}
    for props in props_list:
        cid = props.get("classification", {}).get("id", 0)
        cname = props.get("classification", {}).get("name", f"Class {cid}")
        if cid not in seen_classes:
            seen_classes[cid] = cname

    if seen_classes:
        legend_patches = []
        for cid in sorted(seen_classes):
            color = CLASS_COLORS_HEX[cid] if 0 <= cid < len(CLASS_COLORS_HEX) else "#CCCCCC"
            legend_patches.append(
                plt.Line2D(
                    [0], [0],
                    marker="s", color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=f"{cid}: {seen_classes[cid]}",
                )
            )
        legend = ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=8,
            title="Classes",
            title_fontsize=9,
            framealpha=0.8,
        )

    # --- Title ---
    metadata = geojson.get("metadata", {})
    crop_info = ""
    if "width" in metadata:
        crop_info = (
            f" | Image: {metadata['width']}x{metadata['height']}"
        )
    ax.set_title(
        f"{image_name} | Level {level} ({w}x{h}, ds: {downsample:.0f}x){crop_info}\n"
        f"Annotations: {n_features} features | {len(seen_classes)} classes",
        fontsize=11,
    )
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Invert Y to match image coordinates

    plt.tight_layout()

    if save_path:
        print(f"\nSaving to: {save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print("Done.")
    else:
        print("\nDisplaying... (close window to exit)")
        plt.show()

    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overlay GeoJSON annotations on pyramidal TIFF images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available images interactively
  python visualize_geojson_overlay.py /media/abel/TOSHIBA\\ EXT/export

  # Specific image at level 2
  python visualize_geojson_overlay.py /media/abel/TOSHIBA\\ EXT/export --name B1-01-A-01-001 --level 2

  # Auto-level, save to PNG
  python visualize_geojson_overlay.py /media/abel/TOSHIBA\\ EXT/export --name B1-01-A-01-001 --save overlay.png
        """,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Root export directory (contains images/ and geojson/ subdirectories).",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Base image name (e.g. B1-01-A-01-001). If omitted, shows interactive list.",
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=None,
        help="Pyramid level to load (0 = full resolution). Default: auto (fits in ~4MP).",
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.25,
        help="Fill transparency for polygons (0.0-1.0, default: 0.25).",
    )
    parser.add_argument(
        "--linewidth", "-w",
        type=float,
        default=1.0,
        help="Outline width for polygons (default: 1.0).",
    )
    parser.add_argument(
        "--geojson-subdir",
        type=str,
        default="vector_labels",
        help="Subdirectory for GeoJSON files (default: geojson).",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="Subdirectory for TIFF images (default: images).",
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Save figure to this path instead of displaying interactively.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures (default: 150).",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    visualize(
        dataset_dir=dataset_dir,
        image_name=args.name,
        level=args.level,
        alpha=args.alpha,
        linewidth=args.linewidth,
        geojson_subdir=args.geojson_subdir,
        images_subdir=args.images_subdir,
        save_path=args.save,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
