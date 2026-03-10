#!/usr/bin/env python3
"""
QuPath Export Handler

Class to efficiently handle pyramidal OME-TIFF images and masks exported from QuPath.
Optimized to avoid RAM saturation.

Features:
- Efficient reading using pyramid levels
- Interactive visualization with synchronized zoom/pan
- Lazy loading: only loads what is displayed
- Support for region-based reading

Usage:
    from qupath_handler import QuPathHandler

    handler = QuPathHandler("/path/to/data")
    handler.load_pair("image_name")
    handler.visualize_interactive()
"""

import os
import sys
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

# Agg backend for batch mode (no GUI window)
if "--batch-save" in sys.argv or "--save-all" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider
import tifffile


def load_clinical_data(data_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Load clinical data from data/clinical_diagnosis.txt

    Supports two CSV formats:
    - Legacy: AnonymusCode,Diagnosis,ISUPGradeGroup,Gleasonscore,Scanner
    - New: ANONYMOUS_CODE,PATIENT_NUMBER,AGE,PROSTATE-SPECIFIC_ANTIGEN_(PSA)_LEVEL,
           DIGITAL_RECTAL_EXAM,FINDINGS_IN_PELVIC_MRI,SLIDE_DIAGNOSIS,
           ISUP_Grade_Group_,Gleason_score,Scanner

    Returns:
        Dict mapping image_name -> {diagnosis, isup, gleason, scanner, age, psa, ...}
    """
    clinical_data = {}
    
    # Search for file in several possible locations
    # Priority: specified dir > data subdir > parent dir
    possible_paths = [
        data_dir / "clinical_diagnosis.txt",           # Directly in dataset_dir
        data_dir / "clinical_diagnosis.csv",          # With .csv extension
        data_dir / "data" / "clinical_diagnosis.txt",  # In data/ subdir
        data_dir / "data" / "clinical_diagnosis.csv",  # In data/ with .csv
        data_dir.parent / "clinical_diagnosis.txt",    # In parent dir
        data_dir.parent / "data" / "clinical_diagnosis.txt",  # In parent/data/
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        print(f"[DEBUG] clinical_diagnosis.txt not found")
        print(f"[DEBUG] data_dir = {data_dir}")
        print(f"[DEBUG] Searched in:")
        for p in possible_paths[:4]:  # Show first 4 paths
            print(f"         - {p}")
        return clinical_data

    print(f"[Clinical] Loading from: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both header formats
                code = (
                    row.get('ANONYMOUS_CODE', '') or 
                    row.get('AnonymusCode', '')
                ).strip()
                
                if code:
                    clinical_data[code] = {
                        # Diagnosis
                        'diagnosis': (
                            row.get('SLIDE_DIAGNOSIS', '') or 
                            row.get('Diagnosis', '')
                        ),
                        # ISUP Grade Group
                        'isup': (
                            row.get('ISUP_Grade_Group_', '') or 
                            row.get('ISUPGradeGroup', '')
                        ),
                        # Gleason score
                        'gleason': (
                            row.get('Gleason_score', '') or 
                            row.get('Gleasonscore', '')
                        ),
                        # Scanner
                        'scanner': row.get('Scanner', ''),
                        # New fields (only in new format)
                        'patient_number': row.get('PATIENT_NUMBER', ''),
                        'age': row.get('AGE', ''),
                        'psa': row.get('PROSTATE-SPECIFIC_ANTIGEN_(PSA)_LEVEL', ''),
                        'digital_rectal_exam': row.get('DIGITAL_RECTAL_EXAM', ''),
                        'mri_findings': row.get('FINDINGS_IN_PELVIC_MRI', ''),
                    }
    except Exception:
        pass  # If read fails, return empty dict
    
    return clinical_data

CLASS_NAMES = {
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

CLASS_COLORS_HEX = [
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
    "#4ECDC4",  # 10: Artifact
    "#FFB6B9",  # 11: Seminal vesicle
    "#5BE67D",  # 12: Adipose tissue (green)
    "#61C0BF",  # 13: Normal secretions
    "#8AC6D1",  # 14: Stromal retraction spaces
    "#903749",  # 15: Muscle
    "#E74C3C",  # 16: Foreign body contamination
    "#A8D8EA",  # 17: HGPIN
    "#DDDDDD",  # 18: Calcifications
    "#6A0572",  # 19: Intestinal glands and mucus
    "#FF5722",  # 20: Perineural invasion (PNI)
    "#C0392B",  # 21: Hemorrahage
    "#AB83A1",  # 22: Intraductal carcinoma
    "#53354A",  # 23: Necrosis
    "#E84545",  # 24: Mitosis
    "#BBDED6",  # 25: Nerve ganglion
    "#95E1D3",  # 26: Atypical intraductal proliferation
    "#CC0000",  # 27: Red blood cells
    "#F9ED69",  # 28: Stroma
]

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


CLASS_COLORS_RGB = [hex_to_rgb(c) for c in CLASS_COLORS_HEX]


class PyramidTiff:
    """
    Efficient wrapper for pyramidal TIFF (OME-TIFF).

    Automatically detects pyramid structure:
    - series[0].levels: Levels within a series (standard OME-TIFF)
    - Multiple series: Each series is a level
    - Pages: Each page is a level
    """
    
    def __init__(self, path: str, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.tif = tifffile.TiffFile(path)
        
        # Detect pyramid structure
        self._detect_pyramid_structure()

        # Cache level info
        self._cache_level_info()
        
        if self.verbose:
            self._print_info()
    
    def _detect_pyramid_structure(self) -> None:
        """Detect how pyramid levels are organized."""
        self._pyramid_type = None
        self._levels_source = []
        
        # Option 1: series[0].levels (OME-TIFF with subresolutions)
        if len(self.tif.series) > 0:
            first_series = self.tif.series[0]
            if hasattr(first_series, 'levels') and len(first_series.levels) > 1:
                self._pyramid_type = 'series_levels'
                self._levels_source = first_series.levels
                self.n_levels = len(self._levels_source)
                return
        
        # Option 2: Multiple series (each is a level)
        if len(self.tif.series) > 1:
            # Verify series are decreasing in size
            shapes = [s.shape for s in self.tif.series]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'multiple_series'
                self._levels_source = self.tif.series
                self.n_levels = len(self._levels_source)
                return
        
        # Option 3: Multiple pages
        if len(self.tif.pages) > 1:
            shapes = [p.shape for p in self.tif.pages]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'pages'
                self._levels_source = list(self.tif.pages)
                self.n_levels = len(self._levels_source)
                return
        
        # Fallback: Single level only
        self._pyramid_type = 'single'
        if len(self.tif.series) > 0:
            self._levels_source = [self.tif.series[0]]
        else:
            self._levels_source = [self.tif.pages[0]]
        self.n_levels = 1
    
    def _shapes_are_pyramid(self, shapes: List[Tuple]) -> bool:
        """Check if shapes correspond to a pyramid (decreasing sizes)."""
        if len(shapes) < 2:
            return False
        
        # Extract main size from each shape
        sizes = []
        for shape in shapes:
            # Take the two largest dimensions
            dims = sorted(shape, reverse=True)[:2]
            sizes.append(max(dims))
        
        # Verify they are decreasing
        for i in range(1, len(sizes)):
            if sizes[i] >= sizes[i-1]:
                return False
        return True
    
    def _cache_level_info(self) -> None:
        """Cache info for each level WITHOUT loading data."""
        self.level_info: List[Dict[str, Any]] = []
        base_w, base_h = 0, 0
        
        for i, level_src in enumerate(self._levels_source):
            shape = level_src.shape
            h, w, c = self._parse_shape(shape)
            
            if i == 0:
                ds = 1.0
                base_w, base_h = w, h
            else:
                ds = base_w / w if w > 0 else 1.0
            
            self.level_info.append({
                'index': i,
                'shape': shape,
                'width': w,
                'height': h,
                'channels': c,
                'downsample': ds
            })
    
    def _parse_shape(self, shape: Tuple) -> Tuple[int, int, int]:
        """Interpret shape to get H, W, C."""
        if len(shape) == 2:
            return shape[0], shape[1], 1
        elif len(shape) == 3:
            if shape[0] <= 4:  # (C, H, W)
                return shape[1], shape[2], shape[0]
            else:  # (H, W, C)
                return shape[0], shape[1], shape[2]
        elif len(shape) >= 4:
            # OME: (T, C, Z, Y, X) or similar - Y, X are last
            return shape[-2], shape[-1], shape[1] if shape[1] <= 4 else 1
        return shape[0], 1, 1
    
    def _print_info(self) -> None:
        """Print debug info."""
        print(f"  [PyramidTiff] Type: {self._pyramid_type}")
        print(f"  [PyramidTiff] Levels: {self.n_levels}")
        for info in self.level_info:
            print(f"    Level {info['index']}: {info['width']}x{info['height']} "
                  f"(ds: {info['downsample']:.0f}x, shape: {info['shape']})")
    
    def get_level_for_display(self, max_pixels: int = 4_000_000) -> int:
        """Return the most suitable level given a pixel limit."""
        for info in self.level_info:
            pixels = info['width'] * info['height']
            if pixels <= max_pixels:
                return info['index']
        return self.n_levels - 1
    
    def read_level(self, level: int = 0) -> np.ndarray:
        """Read a complete pyramid level."""
        level = min(level, self.n_levels - 1)
        
        if self.verbose:
            info = self.level_info[level]
            print(f"  [PyramidTiff] Reading level {level}: {info['width']}x{info['height']}")
        
        level_src = self._levels_source[level]
        data = level_src.asarray()
        
        if self.verbose:
            print(f"  [PyramidTiff] Loaded: shape={data.shape}, RAM={data.nbytes/1024/1024:.1f}MB")
        
        return self._normalize_shape(data)
    
    def _normalize_shape(self, data: np.ndarray) -> np.ndarray:
        """Normalize to (H, W) or (H, W, C)."""
        data = np.squeeze(data)
        
        if data.ndim == 2:
            return data
        elif data.ndim == 3:
            if data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
                return np.moveaxis(data, 0, -1)
            return data
        
        # More than 3 dimensions: reduce
        while data.ndim > 3:
            data = data[0]
        return self._normalize_shape(data)
    
    @property
    def base_shape(self) -> Tuple[int, int]:
        """Return (width, height) of base level."""
        return (self.level_info[0]['width'], self.level_info[0]['height'])
    
    def close(self) -> None:
        self.tif.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class QuPathHandler:
    """
    Handler for image/mask pairs exported from QuPath.
    Optimized for efficient visualization without saturating RAM.
    """
    
    def __init__(
        self, 
        data_dir: str,
        images_subdir: str = "images",
        masks_subdir: str = "masks",
        save_resolution: int = 3840,
        output_dir: Optional[str] = None
    ):
        """
        Args:
            data_dir: Base directory with data
            images_subdir: Images subdirectory (or same dir if None)
            masks_subdir: Masks subdirectory (or same dir if None)
            save_resolution: Width in pixels for saving images (default: 3840 = 4K)
            output_dir: Output directory for saved images (default: script_dir/preview)
        """
        self.data_dir = Path(data_dir)
        
        if images_subdir:
            self.images_dir = self.data_dir / images_subdir
        else:
            self.images_dir = self.data_dir
            
        if masks_subdir:
            self.masks_dir = self.data_dir / masks_subdir
        else:
            self.masks_dir = self.data_dir
        
        # Save configuration
        self.save_resolution = save_resolution
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default: script dir / preview
            script_dir = Path(__file__).parent.resolve()
            self.output_dir = script_dir / "preview"
        
        # Current state
        self.current_name: str | None = None
        self.image_tiff: PyramidTiff | None = None
        self.mask_tiff: PyramidTiff | None = None
        self.current_level: int = 0
        
        # Loaded data for current level
        self.image_data: np.ndarray | None = None
        self.mask_data: np.ndarray | None = None
        
        # Clinical data
        self.clinical_data = load_clinical_data(self.data_dir)
        if self.clinical_data:
            print(f"Loaded clinical data for {len(self.clinical_data)} images")
        else:
            print("clinical_diagnosis.txt file not found")

        # Colormap for masks (all classes: 0-28)
        self.mask_cmap = ListedColormap(CLASS_COLORS_HEX)
        self.mask_norm = BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS_HEX) + 0.5, 1), self.mask_cmap.N)
    
    def list_images(self) -> List[str]:
        """List all available images."""
        patterns = ["*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff", "*__mask_multiclass.ome.tif"]
        images = []
        for pattern in patterns:
            images.extend(self.images_dir.glob(pattern))
        
        # Extract base names (without extension or suffixes)
        names = set()
        for img in images:
            name = img.stem
            # Remove common suffixes
            for suffix in [".ome", "__mask_multiclass"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            names.add(name)
        
        return sorted(names)
    
    def _find_file(self, directory: Path, base_name: str, suffixes: List[str]) -> Path | None:
        """Search for a file with different possible names."""
        extensions = [".ome.tif", ".ome.tiff", ".tif", ".tiff"]
        
        for suffix in suffixes:
            for ext in extensions:
                path = directory / f"{base_name}{suffix}{ext}"
                if path.exists():
                    return path
                print(path)
        return None
    
    def load_pair(self, name: str, level: int | None = None) -> None:
        """
        Load an image/mask pair.

        Args:
            name: Base image name (without extension)
            level: Pyramid level to load. None = auto (level that fits in ~4MP)
        """
        # Close previous files
        self.close()
        
        self.current_name = name
        
        # Search for image
        image_path = self._find_file(
            self.images_dir, 
            name, 
            ["", ".ome"]
        )
        
        # Search for mask
        mask_path = self._find_file(
            self.masks_dir, 
            name, 
            ["__mask_multiclass", "_mask", "__mask"]
        )
        
        if image_path is None:
            print(f"Image not found for: {name}")
            print(f"Searched in: {self.images_dir}")
            return

        print(f"Loading image: {image_path.name}")
        self.image_tiff = PyramidTiff(str(image_path))
        
        if mask_path is not None:
            print(f"Loading mask: {mask_path.name}")
            self.mask_tiff = PyramidTiff(str(mask_path))
        else:
            print(f"Mask not found for: {name}")
            self.mask_tiff = None
        
        # Determine level to load
        if level is None:
            level = self.image_tiff.get_level_for_display(max_pixels=4_000_000)
        
        self._load_level(level)
    
    def _load_level(self, level: int) -> None:
        """Load a specific level into memory."""
        import gc
        
        if self.image_tiff is None:
            return
        
        level = min(level, self.image_tiff.n_levels - 1)
        self.current_level = level
        
        print(f"\nLoading level {level}...")

        # Free memory from previous level BEFORE loading new one
        self.image_data = None
        self.mask_data = None
        gc.collect()
        
        # Load image
        self.image_data = self.image_tiff.read_level(level)
        info = self.image_tiff.level_info[level]
        print(f"  Image: {info['width']} x {info['height']} (downsample: {info['downsample']:.1f}x)")
        print(f"  Image RAM: {self.image_data.nbytes / 1024 / 1024:.1f} MB")

        # Load mask at same level if exists
        if self.mask_tiff is not None:
            mask_level = min(level, self.mask_tiff.n_levels - 1)
            self.mask_data = self.mask_tiff.read_level(mask_level)
            
            # Ensure 2D for mask
            if self.mask_data.ndim == 3:
                self.mask_data = self.mask_data[:, :, 0]
            
            print(f"  Mask: {self.mask_data.shape}")
            print(f"  Mask RAM: {self.mask_data.nbytes / 1024 / 1024:.1f} MB")
        
        gc.collect()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for current pair."""
        if self.image_tiff is None:
            return {}
        
        return {
            'name': self.current_name,
            'current_level': self.current_level,
            'image_levels': [
                {
                    'level': i,
                    'size': f"{info['width']} x {info['height']}",
                    'downsample': info['downsample']
                }
                for i, info in enumerate(self.image_tiff.level_info)
            ],
            'mask_levels': [
                {
                    'level': i,
                    'size': f"{info['width']} x {info['height']}",
                    'downsample': info['downsample']
                }
                for i, info in enumerate(self.mask_tiff.level_info)
            ] if self.mask_tiff else [],
            'base_size': self.image_tiff.base_shape,
        }
    
    def get_data(self) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Return (image, mask) for current level."""
        return self.image_data, self.mask_data
    
    def get_clinical_info(self, name: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Get clinical information for an image.

        Args:
            name: Image name (without extension). If None, uses current_name.

        Returns:
            Dict with diagnosis, isup, gleason, scanner, or None if not found.
        """
        if name is None:
            name = self.current_name
        if name is None:
            return None
        return self.clinical_data.get(name)
    
    def format_clinical_title(self, name: Optional[str] = None) -> str:
        """
        Generate a string with clinical info for use in titles.

        Returns:
            String with clinical info in two lines:
            Line 1: Diagnosis | Age | PSA | MRI
            Line 2: ISUP | Gleason | Scanner
        """
        if name is None:
            name = self.current_name
        info = self.get_clinical_info(name)
        print(f"[DEBUG] format_clinical_title: name={name}, info={info}")
        if info is None:
            return ""
        
        # First line: diagnosis and patient data
        line1_parts = []
        if info.get('diagnosis'):
            line1_parts.append(f"{info['diagnosis']}")
        if info.get('age'):
            line1_parts.append(f"Age: {info['age']}")
        if info.get('psa'):
            line1_parts.append(f"PSA: {info['psa']}")
        if info.get('mri_findings'):
            line1_parts.append(f"MRI: {info['mri_findings']}")
        
        # Second line: classification and scanner
        line2_parts = []
        if info.get('isup') and info['isup'] != '0':
            line2_parts.append(f"ISUP: {info['isup']}")
        if info.get('gleason') and info['gleason'] != '0':
            line2_parts.append(f"Gleason: {info['gleason']}")
        if info.get('scanner'):
            line2_parts.append(f"Scanner: {info['scanner']}")
        
        # Combine lines
        lines = []
        if line1_parts:
            lines.append(" | ".join(line1_parts))
        if line2_parts:
            lines.append(" | ".join(line2_parts))
        
        return "\n".join(lines)
    
    def _save_figure(self, fig: plt.Figure, show_legend: bool = True) -> None:
        """
        Save current figure as high-resolution PNG.

        Args:
            fig: Matplotlib figure to save
            show_legend: Whether to include full legend
        """
        if self.current_name is None:
            print("  No image loaded to save.")
            return
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: image name
        filename = f"{self.current_name}.png"
        filepath = self.output_dir / filename
        
        # Calculate DPI to reach desired resolution
        fig_width_inches = fig.get_figwidth()
        target_dpi = self.save_resolution / fig_width_inches
        
        # Save with high quality
        print(f"  Saving: {filepath}")
        print(f"  Resolution: {self.save_resolution}px (DPI: {target_dpi:.0f})")
        
        fig.savefig(
            filepath,
            dpi=target_dpi,
            bbox_inches='tight',
            pad_inches=0.2,
            facecolor='white',
            edgecolor='none',
            format='png'
        )
        
        # Verify final size
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                w, h = img.size
                file_size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  Saved: {w}x{h}px ({file_size_mb:.1f} MB)")
        except ImportError:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Saved: {file_size_mb:.1f} MB")
    
    def visualize(self, show_legend: bool = True, save_only: bool = False) -> None:
        """
        Visualization with synchronized zoom/pan between image and mask.
        Uses matplotlib zoom tool to explore.

        Args:
            show_legend: Whether to show class legend
            save_only: If True, save PNG and close without showing window (batch mode)

        Controls (when save_only=False):
            - S: Save high-resolution PNG
            - Q: Close window
        """
        if self.image_data is None:
            print("No data loaded. Use load_pair() first.")
            return
        
        has_mask = self.mask_data is not None
        n_cols = 2 if has_mask else 1
        
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
        if n_cols == 1:
            axes = [axes]
        
        ax_img = axes[0]
        ax_mask = axes[1] if has_mask else None
        
        # Title with clinical info
        level_info = self.image_tiff.level_info[self.current_level]
        title = f"{self.current_name} | Level {self.current_level} | {level_info['width']}x{level_info['height']} (ds: {level_info['downsample']:.0f}x)"
        clinical_str = self.format_clinical_title()
        if clinical_str:
            title = f"{title}\n{clinical_str}"
        fig.suptitle(title, fontsize=11)
        
        # Imagen
        ax_img.imshow(self.image_data)
        ax_img.set_title("Image WSI")
        ax_img.axis('off')
        
        # Máscara
        if has_mask and ax_mask is not None:
            ax_mask.imshow(
                self.mask_data,
                cmap=self.mask_cmap,
                norm=self.mask_norm,
                interpolation='nearest'
            )
            ax_mask.set_title("Classification mask")
            ax_mask.axis('off')
            
            if show_legend:
                unique_classes = np.unique(self.mask_data)
                legend_patches = [
                    Patch(
                        facecolor=CLASS_COLORS_HEX[c],
                        edgecolor='black',
                        label=f"{c}: {CLASS_NAMES.get(c, '?')}"
                    )
                    for c in sorted(unique_classes) if c in CLASS_NAMES
                ]
                
                ax_mask.legend(
                    handles=legend_patches,
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=8
                )
            
            # Synchronize zoom/pan between both axes
            self._syncing = False
            
            def sync_from_img(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    ax_mask.set_xlim(ax_img.get_xlim())
                    ax_mask.set_ylim(ax_img.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            
            def sync_from_mask(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    ax_img.set_xlim(ax_mask.get_xlim())
                    ax_img.set_ylim(ax_mask.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            
            ax_img.callbacks.connect('xlim_changed', sync_from_img)
            ax_img.callbacks.connect('ylim_changed', sync_from_img)
            ax_mask.callbacks.connect('xlim_changed', sync_from_mask)
            ax_mask.callbacks.connect('ylim_changed', sync_from_mask)
        
        # Handler to save with S key (only if not save_only)
        if not save_only:
            def on_key(event):
                if event.key == 's':
                    self._save_figure(fig, show_legend=show_legend)
            fig.canvas.mpl_connect('key_press_event', on_key)
            print("  [Controls] S: save PNG | Q: close")
        
        plt.tight_layout()
        if save_only:
            self._save_figure(fig, show_legend=show_legend)
            plt.close(fig)
        else:
            plt.show()
    
    def visualize_interactive(self) -> None:
        """
        Interactive visualization with level slider.
        Allows changing resolution level in real time.
        """
        if self.image_tiff is None:
            print("No data loaded. Use load_pair() first.")
            return
        
        has_mask = self.mask_data is not None
        n_cols = 2 if has_mask else 1
        
        # Create figure with space for slider
        fig = plt.figure(figsize=(7 * n_cols, 8))
        
        # Axes for images
        ax_img = fig.add_axes([0.05, 0.15, 0.4 if has_mask else 0.9, 0.75])
        ax_mask = fig.add_axes([0.55, 0.15, 0.4, 0.75]) if has_mask else None
        
        # Axis for slider
        ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03])
        
        # Level slider
        n_levels = self.image_tiff.n_levels
        slider = Slider(
            ax_slider,
            'Level',
            0,
            n_levels - 1,
            valinit=self.current_level,
            valstep=1
        )
        
        # Show initial images
        img_display = ax_img.imshow(self.image_data)
        ax_img.set_title("Image")
        ax_img.axis('off')
        
        mask_display = None
        if has_mask and ax_mask is not None:
            mask_display = ax_mask.imshow(
                self.mask_data,
                cmap=self.mask_cmap,
                norm=self.mask_norm,
                interpolation='nearest'
            )
            ax_mask.set_title("Mask")
            ax_mask.axis('off')
        
        def update_title():
            info = self.image_tiff.level_info[self.current_level]
            title = (
                f"{self.current_name} | Level {self.current_level} | "
                f"{info['width']}x{info['height']} (ds: {info['downsample']:.0f}x)"
            )
            clinical_str = self.format_clinical_title()
            if clinical_str:
                title = f"{title}\n{clinical_str}"
            fig.suptitle(title, fontsize=11)
        
        update_title()
        
        def on_slider_change(val):
            level = int(val)
            if level != self.current_level:
                self._load_level(level)
                img_display.set_data(self.image_data)
                img_display.set_extent([0, self.image_data.shape[1], self.image_data.shape[0], 0])
                ax_img.set_xlim(0, self.image_data.shape[1])
                ax_img.set_ylim(self.image_data.shape[0], 0)
                
                if mask_display is not None and self.mask_data is not None:
                    mask_display.set_data(self.mask_data)
                    mask_display.set_extent([0, self.mask_data.shape[1], self.mask_data.shape[0], 0])
                    ax_mask.set_xlim(0, self.mask_data.shape[1])
                    ax_mask.set_ylim(self.mask_data.shape[0], 0)
                
                update_title()
                fig.canvas.draw_idle()
        
        slider.on_changed(on_slider_change)
        
        # Synchronize zoom/pan between both axes
        if ax_mask is not None:
            # Flag to avoid infinite recursion
            self._syncing = False
            
            def sync_from_img(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    ax_mask.set_xlim(ax_img.get_xlim())
                    ax_mask.set_ylim(ax_img.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            
            def sync_from_mask(event_ax):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    ax_img.set_xlim(ax_mask.get_xlim())
                    ax_img.set_ylim(ax_mask.get_ylim())
                    fig.canvas.draw_idle()
                finally:
                    self._syncing = False
            
            ax_img.callbacks.connect('xlim_changed', sync_from_img)
            ax_img.callbacks.connect('ylim_changed', sync_from_img)
            ax_mask.callbacks.connect('xlim_changed', sync_from_mask)
            ax_mask.callbacks.connect('ylim_changed', sync_from_mask)
        
        # Handler to save with S key
        def on_key(event):
            if event.key == 's':
                self._save_figure(fig, show_legend=True)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("  [Controls] S: save PNG | Q: close | Slider: change level")
        plt.show()
    
    def change_level(self, level: int) -> None:
        """Change to specified level."""
        if self.image_tiff is None:
            print("No data loaded.")
            return
        self._load_level(level)
    
    def close(self) -> None:
        """Close open files and free memory."""
        if self.image_tiff is not None:
            self.image_tiff.close()
            self.image_tiff = None
        
        if self.mask_tiff is not None:
            self.mask_tiff.close()
            self.mask_tiff = None
        
        self.image_data = None
        self.mask_data = None
        self.current_name = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def main():
    """Iterate over all images in the dataset and visualize them."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="QuPath image and mask visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all images in the dataset
  python qupath_handler.py /path/to/data

  # Specify resolution level
  python qupath_handler.py /path/to/data --level 2

  # Save screenshots in 8K
  python qupath_handler.py /path/to/data --save-resolution 7680

  # Batch mode: save all images without opening windows
  python qupath_handler.py /path/to/data --batch-save

Expected directory structure:
  data_dir/
    images/
      image1.ome.tif
      image2.ome.tif
    masks/
      image1__mask_multiclass.ome.tif
      image2__mask_multiclass.ome.tif

Controls:
  S: Save high-resolution PNG
  Q: Close window and go to next
        """
    )
    
    parser.add_argument("data_dir", help="Dataset directory with images/ and masks/")
    parser.add_argument("--level", "-l", type=int, default=None, help="Pyramid level (default: auto)")
    parser.add_argument("--images-dir", default="images", help="Images subdirectory")
    parser.add_argument("--masks-dir", default="masks", help="Masks subdirectory")
    parser.add_argument("--save-resolution", "-r", type=int, default=3840,
                        help="Width in pixels for saving images (default: 3840 = 4K)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for screenshots (default: script_dir/preview)")
    parser.add_argument("--batch-save", "--save-all", dest="batch_save", action="store_true",
                        help="Iterate all images and save PNG without opening windows")
    
    args = parser.parse_args()
    
    handler = QuPathHandler(
        args.data_dir,
        images_subdir=args.images_dir,
        masks_subdir=args.masks_dir,
        save_resolution=args.save_resolution,
        output_dir=args.output_dir
    )
    
    images = handler.list_images()
    
    if not images:
        print("No images found.")
        return

    print(f"\nFound {len(images)} images")
    if args.batch_save:
        print("Batch mode: saving PNG without opening windows.\n")
    else:
        print("Close each window to proceed to the next.\n")
    
    for i, name in enumerate(images):
        print(f"[{i+1}/{len(images)}] {name}")
        
        try:
            handler.load_pair(name, level=args.level)
            
            meta = handler.get_metadata()
            print(f"  Base size: {meta.get('base_size')}")
            print(f"  Loaded level: {meta.get('current_level')}")

            # Show clinical info if available
            clinical = handler.get_clinical_info(name)
            if clinical:
                print(f"  Diagnosis: {clinical.get('diagnosis', 'N/A')}")
                # Patient data
                age = clinical.get('age', '')
                psa = clinical.get('psa', '')
                if age or psa:
                    patient_info = []
                    if age:
                        patient_info.append(f"Age: {age}")
                    if psa:
                        patient_info.append(f"PSA: {psa}")
                    print(f"  {' | '.join(patient_info)}")
                # MRI
                mri = clinical.get('mri_findings', '')
                if mri:
                    print(f"  MRI: {mri}")
                # Classification
                isup = clinical.get('isup', '')
                gleason = clinical.get('gleason', '')
                if isup and isup != '0':
                    print(f"  ISUP Grade: {isup} | Gleason: {gleason}")
                print(f"  Scanner: {clinical.get('scanner', 'N/A')}")
            
            if args.batch_save:
                handler.visualize(save_only=True)
            else:
                handler.visualize()
            
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            handler.close()
    
    print("\nVisualization completed.")


if __name__ == "__main__":
    main()
