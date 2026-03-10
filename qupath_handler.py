#!/usr/bin/env python3
"""
QuPath Export Handler

Clase para manejar eficientemente las imágenes y máscaras OME-TIFF piramidales
exportadas desde QuPath. Optimizada para no saturar RAM.

Características:
- Lectura eficiente usando niveles de pirámide
- Visualización interactiva con zoom/pan sincronizado
- Carga lazy: solo carga lo que se muestra
- Soporte para lectura por regiones

Uso:
    from qupath_handler import QuPathHandler
    
    handler = QuPathHandler("/ruta/a/datos")
    handler.load_pair("nombre_imagen")
    handler.visualize_interactive()
"""

import os
import sys
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

# Backend Agg para modo batch (sin ventana gráfica)
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
    Carga datos clínicos desde data/clinical_diagnosis.txt
    
    Soporta dos formatos de CSV:
    - Antiguo: AnonymusCode,Diagnosis,ISUPGradeGroup,Gleasonscore,Scanner
    - Nuevo: ANONYMOUS_CODE,PATIENT_NUMBER,AGE,PROSTATE-SPECIFIC_ANTIGEN_(PSA)_LEVEL,
             DIGITAL_RECTAL_EXAM,FINDINGS_IN_PELVIC_MRI,SLIDE_DIAGNOSIS,
             ISUP_Grade_Group_,Gleason_score,Scanner
    
    Returns:
        Dict mapping image_name -> {diagnosis, isup, gleason, scanner, age, psa, ...}
    """
    clinical_data = {}
    
    # Buscar el archivo en varias ubicaciones posibles
    # Prioridad: directorio indicado > subdirectorio data > directorio padre
    possible_paths = [
        data_dir / "clinical_diagnosis.txt",           # Directamente en dataset_dir
        data_dir / "clinical_diagnosis.csv",           # Con extensión .csv
        data_dir / "data" / "clinical_diagnosis.txt",  # En subdirectorio data/
        data_dir / "data" / "clinical_diagnosis.csv",  # En data/ con .csv
        data_dir.parent / "clinical_diagnosis.txt",    # En directorio padre
        data_dir.parent / "data" / "clinical_diagnosis.txt",  # En padre/data/
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        print(f"[DEBUG] No se encontró clinical_diagnosis.txt")
        print(f"[DEBUG] data_dir = {data_dir}")
        print(f"[DEBUG] Buscado en:")
        for p in possible_paths[:4]:  # Mostrar las primeras 4 rutas
            print(f"         - {p}")
        return clinical_data
    
    print(f"[Clinical] Cargando desde: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Soportar ambos formatos de cabecera
                code = (
                    row.get('ANONYMOUS_CODE', '') or 
                    row.get('AnonymusCode', '')
                ).strip()
                
                if code:
                    clinical_data[code] = {
                        # Diagnóstico
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
                        # Nuevos campos (solo en formato nuevo)
                        'patient_number': row.get('PATIENT_NUMBER', ''),
                        'age': row.get('AGE', ''),
                        'psa': row.get('PROSTATE-SPECIFIC_ANTIGEN_(PSA)_LEVEL', ''),
                        'digital_rectal_exam': row.get('DIGITAL_RECTAL_EXAM', ''),
                        'mri_findings': row.get('FINDINGS_IN_PELVIC_MRI', ''),
                    }
    except Exception:
        pass  # Si falla la lectura, devolver diccionario vacío
    
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
    "#000000",  # 0: Background (negro)
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
    "#5BE67D",  # 12: Adipose tissue (verde)
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
    """Convierte color hex a RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


CLASS_COLORS_RGB = [hex_to_rgb(c) for c in CLASS_COLORS_HEX]


class PyramidTiff:
    """
    Wrapper eficiente para TIFF piramidal (OME-TIFF).
    
    Detecta automáticamente la estructura de la pirámide:
    - series[0].levels: Niveles dentro de una serie (OME-TIFF estándar)
    - Múltiples series: Cada serie es un nivel
    - Páginas: Cada página es un nivel
    """
    
    def __init__(self, path: str, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.tif = tifffile.TiffFile(path)
        
        # Detectar estructura de la pirámide
        self._detect_pyramid_structure()
        
        # Cachear info de niveles
        self._cache_level_info()
        
        if self.verbose:
            self._print_info()
    
    def _detect_pyramid_structure(self) -> None:
        """Detecta cómo están organizados los niveles de la pirámide."""
        self._pyramid_type = None
        self._levels_source = []
        
        # Opción 1: series[0].levels (OME-TIFF con subresoluciones)
        if len(self.tif.series) > 0:
            first_series = self.tif.series[0]
            if hasattr(first_series, 'levels') and len(first_series.levels) > 1:
                self._pyramid_type = 'series_levels'
                self._levels_source = first_series.levels
                self.n_levels = len(self._levels_source)
                return
        
        # Opción 2: Múltiples series (cada una es un nivel)
        if len(self.tif.series) > 1:
            # Verificar que las series son de tamaño decreciente
            shapes = [s.shape for s in self.tif.series]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'multiple_series'
                self._levels_source = self.tif.series
                self.n_levels = len(self._levels_source)
                return
        
        # Opción 3: Múltiples páginas
        if len(self.tif.pages) > 1:
            shapes = [p.shape for p in self.tif.pages]
            if self._shapes_are_pyramid(shapes):
                self._pyramid_type = 'pages'
                self._levels_source = list(self.tif.pages)
                self.n_levels = len(self._levels_source)
                return
        
        # Fallback: Solo un nivel
        self._pyramid_type = 'single'
        if len(self.tif.series) > 0:
            self._levels_source = [self.tif.series[0]]
        else:
            self._levels_source = [self.tif.pages[0]]
        self.n_levels = 1
    
    def _shapes_are_pyramid(self, shapes: List[Tuple]) -> bool:
        """Verifica si las shapes corresponden a una pirámide (tamaños decrecientes)."""
        if len(shapes) < 2:
            return False
        
        # Extraer el tamaño principal de cada shape
        sizes = []
        for shape in shapes:
            # Tomar las dos dimensiones más grandes
            dims = sorted(shape, reverse=True)[:2]
            sizes.append(max(dims))
        
        # Verificar que son decrecientes
        for i in range(1, len(sizes)):
            if sizes[i] >= sizes[i-1]:
                return False
        return True
    
    def _cache_level_info(self) -> None:
        """Cachea información de cada nivel SIN cargar datos."""
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
        """Interpreta shape para obtener H, W, C."""
        if len(shape) == 2:
            return shape[0], shape[1], 1
        elif len(shape) == 3:
            if shape[0] <= 4:  # (C, H, W)
                return shape[1], shape[2], shape[0]
            else:  # (H, W, C)
                return shape[0], shape[1], shape[2]
        elif len(shape) >= 4:
            # OME: (T, C, Z, Y, X) o similar - Y, X son las últimas
            return shape[-2], shape[-1], shape[1] if shape[1] <= 4 else 1
        return shape[0], 1, 1
    
    def _print_info(self) -> None:
        """Imprime información de debug."""
        print(f"  [PyramidTiff] Tipo: {self._pyramid_type}")
        print(f"  [PyramidTiff] Niveles: {self.n_levels}")
        for info in self.level_info:
            print(f"    Nivel {info['index']}: {info['width']}x{info['height']} "
                  f"(ds: {info['downsample']:.0f}x, shape: {info['shape']})")
    
    def get_level_for_display(self, max_pixels: int = 4_000_000) -> int:
        """Devuelve el nivel más adecuado dado un límite de píxeles."""
        for info in self.level_info:
            pixels = info['width'] * info['height']
            if pixels <= max_pixels:
                return info['index']
        return self.n_levels - 1
    
    def read_level(self, level: int = 0) -> np.ndarray:
        """Lee un nivel completo de la pirámide."""
        level = min(level, self.n_levels - 1)
        
        if self.verbose:
            info = self.level_info[level]
            print(f"  [PyramidTiff] Leyendo nivel {level}: {info['width']}x{info['height']}")
        
        level_src = self._levels_source[level]
        data = level_src.asarray()
        
        if self.verbose:
            print(f"  [PyramidTiff] Cargado: shape={data.shape}, RAM={data.nbytes/1024/1024:.1f}MB")
        
        return self._normalize_shape(data)
    
    def _normalize_shape(self, data: np.ndarray) -> np.ndarray:
        """Normaliza a (H, W) o (H, W, C)."""
        data = np.squeeze(data)
        
        if data.ndim == 2:
            return data
        elif data.ndim == 3:
            if data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
                return np.moveaxis(data, 0, -1)
            return data
        
        # Más de 3 dimensiones: reducir
        while data.ndim > 3:
            data = data[0]
        return self._normalize_shape(data)
    
    @property
    def base_shape(self) -> Tuple[int, int]:
        """Devuelve (width, height) del nivel base."""
        return (self.level_info[0]['width'], self.level_info[0]['height'])
    
    def close(self) -> None:
        self.tif.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class QuPathHandler:
    """
    Manejador para pares imagen/máscara exportados desde QuPath.
    Optimizado para visualización eficiente sin saturar RAM.
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
            data_dir: Directorio base con los datos
            images_subdir: Subdirectorio de imágenes (o mismo dir si es None)
            masks_subdir: Subdirectorio de máscaras (o mismo dir si es None)
            save_resolution: Ancho en píxeles para guardar imágenes (default: 3840 = 4K)
            output_dir: Directorio de salida para imágenes guardadas (default: script_dir/figures)
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
        
        # Configuración de guardado
        self.save_resolution = save_resolution
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Por defecto: directorio del script / figures
            script_dir = Path(__file__).parent.resolve()
            self.output_dir = script_dir / "preview"
        
        # Estado actual
        self.current_name: str | None = None
        self.image_tiff: PyramidTiff | None = None
        self.mask_tiff: PyramidTiff | None = None
        self.current_level: int = 0
        
        # Datos cargados del nivel actual
        self.image_data: np.ndarray | None = None
        self.mask_data: np.ndarray | None = None
        
        # Datos clínicos
        self.clinical_data = load_clinical_data(self.data_dir)
        if self.clinical_data:
            print(f"Cargados datos clínicos para {len(self.clinical_data)} imágenes")
        else:
            print("No se encontró archivo clinical_diagnosis.txt")
        
        # Colormap para máscaras (todas las clases: 0-28)
        self.mask_cmap = ListedColormap(CLASS_COLORS_HEX)
        self.mask_norm = BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS_HEX) + 0.5, 1), self.mask_cmap.N)
    
    def list_images(self) -> List[str]:
        """Lista todas las imágenes disponibles."""
        patterns = ["*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff", "*__mask_multiclass.ome.tif"]
        images = []
        for pattern in patterns:
            images.extend(self.images_dir.glob(pattern))
        
        # Extraer nombres base (sin extensión ni sufijos)
        names = set()
        for img in images:
            name = img.stem
            # Quitar sufijos comunes
            for suffix in [".ome", "__mask_multiclass"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            names.add(name)
        
        return sorted(names)
    
    def _find_file(self, directory: Path, base_name: str, suffixes: List[str]) -> Path | None:
        """Busca un archivo con diferentes posibles nombres."""
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
        Carga un par imagen/máscara.
        
        Args:
            name: Nombre base de la imagen (sin extensión)
            level: Nivel de pirámide a cargar. None = auto (nivel que quepa en ~4MP)
        """
        # Cerrar archivos anteriores
        self.close()
        
        self.current_name = name
        
        # Buscar imagen
        image_path = self._find_file(
            self.images_dir, 
            name, 
            ["", ".ome"]
        )
        
        # Buscar máscara
        mask_path = self._find_file(
            self.masks_dir, 
            name, 
            ["__mask_multiclass", "_mask", "__mask"]
        )
        
        if image_path is None:
            print(f"No se encontró imagen para: {name}")
            print(f"Buscado en: {self.images_dir}")
            return
        
        print(f"Cargando imagen: {image_path.name}")
        self.image_tiff = PyramidTiff(str(image_path))
        
        if mask_path is not None:
            print(f"Cargando máscara: {mask_path.name}")
            self.mask_tiff = PyramidTiff(str(mask_path))
        else:
            print(f"No se encontró máscara para: {name}")
            self.mask_tiff = None
        
        # Determinar nivel a cargar
        if level is None:
            level = self.image_tiff.get_level_for_display(max_pixels=4_000_000)
        
        self._load_level(level)
    
    def _load_level(self, level: int) -> None:
        """Carga un nivel específico en memoria."""
        import gc
        
        if self.image_tiff is None:
            return
        
        level = min(level, self.image_tiff.n_levels - 1)
        self.current_level = level
        
        print(f"\nCargando nivel {level}...")
        
        # Liberar memoria del nivel anterior ANTES de cargar el nuevo
        self.image_data = None
        self.mask_data = None
        gc.collect()
        
        # Cargar imagen
        self.image_data = self.image_tiff.read_level(level)
        info = self.image_tiff.level_info[level]
        print(f"  Imagen: {info['width']} x {info['height']} (downsample: {info['downsample']:.1f}x)")
        print(f"  RAM imagen: {self.image_data.nbytes / 1024 / 1024:.1f} MB")
        
        # Cargar máscara al mismo nivel si existe
        if self.mask_tiff is not None:
            mask_level = min(level, self.mask_tiff.n_levels - 1)
            self.mask_data = self.mask_tiff.read_level(mask_level)
            
            # Asegurar 2D para máscara
            if self.mask_data.ndim == 3:
                self.mask_data = self.mask_data[:, :, 0]
            
            print(f"  Máscara: {self.mask_data.shape}")
            print(f"  RAM máscara: {self.mask_data.nbytes / 1024 / 1024:.1f} MB")
        
        gc.collect()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve metadatos del par actual."""
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
        """Devuelve (imagen, máscara) del nivel actual."""
        return self.image_data, self.mask_data
    
    def get_clinical_info(self, name: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Obtiene información clínica para una imagen.
        
        Args:
            name: Nombre de la imagen (sin extensión). Si None, usa current_name.
        
        Returns:
            Dict con diagnosis, isup, gleason, scanner, o None si no encontrado.
        """
        if name is None:
            name = self.current_name
        if name is None:
            return None
        return self.clinical_data.get(name)
    
    def format_clinical_title(self, name: Optional[str] = None) -> str:
        """
        Genera una cadena con información clínica para usar en títulos.
        
        Returns:
            String con información clínica en dos líneas:
            Línea 1: Diagnosis | Age | PSA | MRI
            Línea 2: ISUP | Gleason | Scanner
        """
        if name is None:
            name = self.current_name
        info = self.get_clinical_info(name)
        print(f"[DEBUG] format_clinical_title: name={name}, info={info}")
        if info is None:
            return ""
        
        # Primera línea: diagnóstico y datos del paciente
        line1_parts = []
        if info.get('diagnosis'):
            line1_parts.append(f"{info['diagnosis']}")
        if info.get('age'):
            line1_parts.append(f"Age: {info['age']}")
        if info.get('psa'):
            line1_parts.append(f"PSA: {info['psa']}")
        if info.get('mri_findings'):
            line1_parts.append(f"MRI: {info['mri_findings']}")
        
        # Segunda línea: clasificación y scanner
        line2_parts = []
        if info.get('isup') and info['isup'] != '0':
            line2_parts.append(f"ISUP: {info['isup']}")
        if info.get('gleason') and info['gleason'] != '0':
            line2_parts.append(f"Gleason: {info['gleason']}")
        if info.get('scanner'):
            line2_parts.append(f"Scanner: {info['scanner']}")
        
        # Combinar líneas
        lines = []
        if line1_parts:
            lines.append(" | ".join(line1_parts))
        if line2_parts:
            lines.append(" | ".join(line2_parts))
        
        return "\n".join(lines)
    
    def _save_figure(self, fig: plt.Figure, show_legend: bool = True) -> None:
        """
        Guarda la figura actual como PNG en alta resolución.
        
        Args:
            fig: Figura de matplotlib a guardar
            show_legend: Si incluir la leyenda completa
        """
        if self.current_name is None:
            print("  No hay imagen cargada para guardar.")
            return
        
        # Crear directorio de salida si no existe
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nombre del archivo: nombre de la imagen
        filename = f"{self.current_name}.png"
        filepath = self.output_dir / filename
        
        # Calcular DPI para alcanzar la resolución deseada
        fig_width_inches = fig.get_figwidth()
        target_dpi = self.save_resolution / fig_width_inches
        
        # Guardar con alta calidad
        print(f"  Guardando: {filepath}")
        print(f"  Resolución: {self.save_resolution}px (DPI: {target_dpi:.0f})")
        
        fig.savefig(
            filepath,
            dpi=target_dpi,
            bbox_inches='tight',
            pad_inches=0.2,
            facecolor='white',
            edgecolor='none',
            format='png'
        )
        
        # Verificar tamaño final
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                w, h = img.size
                file_size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  Guardado: {w}x{h}px ({file_size_mb:.1f} MB)")
        except ImportError:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Guardado: {file_size_mb:.1f} MB")
    
    def visualize(self, show_legend: bool = True, save_only: bool = False) -> None:
        """
        Visualización con zoom/pan sincronizado entre imagen y máscara.
        Usa la herramienta de zoom de matplotlib para explorar.
        
        Args:
            show_legend: Si mostrar leyenda de clases
            save_only: Si True, guarda PNG y cierra sin mostrar ventana (modo batch)
        
        Controles (cuando save_only=False):
            - S: Guardar imagen PNG en alta resolución
            - Q: Cerrar ventana
        """
        if self.image_data is None:
            print("No hay datos cargados. Usa load_pair() primero.")
            return
        
        has_mask = self.mask_data is not None
        n_cols = 2 if has_mask else 1
        
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
        if n_cols == 1:
            axes = [axes]
        
        ax_img = axes[0]
        ax_mask = axes[1] if has_mask else None
        
        # Título con información clínica
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
            
            # Sincronizar zoom/pan entre los dos axes
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
        
        # Handler para guardar con tecla S (solo si no es save_only)
        if not save_only:
            def on_key(event):
                if event.key == 's':
                    self._save_figure(fig, show_legend=show_legend)
            fig.canvas.mpl_connect('key_press_event', on_key)
            print("  [Controles] S: guardar PNG | Q: cerrar")
        
        plt.tight_layout()
        if save_only:
            self._save_figure(fig, show_legend=show_legend)
            plt.close(fig)
        else:
            plt.show()
    
    def visualize_interactive(self) -> None:
        """
        Visualización interactiva con slider de nivel.
        Permite cambiar el nivel de resolución en tiempo real.
        """
        if self.image_tiff is None:
            print("No hay datos cargados. Usa load_pair() primero.")
            return
        
        has_mask = self.mask_data is not None
        n_cols = 2 if has_mask else 1
        
        # Crear figura con espacio para slider
        fig = plt.figure(figsize=(7 * n_cols, 8))
        
        # Axes para las imágenes
        ax_img = fig.add_axes([0.05, 0.15, 0.4 if has_mask else 0.9, 0.75])
        ax_mask = fig.add_axes([0.55, 0.15, 0.4, 0.75]) if has_mask else None
        
        # Axis para slider
        ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03])
        
        # Slider de nivel
        n_levels = self.image_tiff.n_levels
        slider = Slider(
            ax_slider,
            'Nivel',
            0,
            n_levels - 1,
            valinit=self.current_level,
            valstep=1
        )
        
        # Mostrar imágenes iniciales
        img_display = ax_img.imshow(self.image_data)
        ax_img.set_title("Imagen")
        ax_img.axis('off')
        
        mask_display = None
        if has_mask and ax_mask is not None:
            mask_display = ax_mask.imshow(
                self.mask_data,
                cmap=self.mask_cmap,
                norm=self.mask_norm,
                interpolation='nearest'
            )
            ax_mask.set_title("Máscara")
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
        
        # Sincronizar zoom/pan entre los dos axes
        if ax_mask is not None:
            # Flag para evitar recursión infinita
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
        
        # Handler para guardar con tecla S
        def on_key(event):
            if event.key == 's':
                self._save_figure(fig, show_legend=True)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("  [Controles] S: guardar PNG | Q: cerrar | Slider: cambiar nivel")
        plt.show()
    
    def change_level(self, level: int) -> None:
        """Cambia al nivel especificado."""
        if self.image_tiff is None:
            print("No hay datos cargados.")
            return
        self._load_level(level)
    
    def close(self) -> None:
        """Cierra los archivos abiertos y libera memoria."""
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
    """Recorre todas las imágenes del dataset y las visualiza."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualizador de imágenes y máscaras QuPath",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Visualizar todas las imágenes del dataset
  python qupath_handler.py /ruta/datos
  
  # Especificar nivel de resolución
  python qupath_handler.py /ruta/datos --level 2
  
  # Guardar screenshots en 8K
  python qupath_handler.py /ruta/datos --save-resolution 7680
  
  # Modo batch: guardar todas las imágenes sin abrir ventanas
  python qupath_handler.py /ruta/datos --batch-save

Estructura de directorios esperada:
  data_dir/
    images/
      imagen1.ome.tif
      imagen2.ome.tif
    masks/
      imagen1__mask_multiclass.ome.tif
      imagen2__mask_multiclass.ome.tif

Controles:
  S: Guardar PNG en alta resolución
  Q: Cerrar ventana y pasar a siguiente
        """
    )
    
    parser.add_argument("data_dir", help="Directorio del dataset con images/ y masks/")
    parser.add_argument("--level", "-l", type=int, default=None, help="Nivel de pirámide (default: auto)")
    parser.add_argument("--images-dir", default="images", help="Subdirectorio de imágenes")
    parser.add_argument("--masks-dir", default="masks", help="Subdirectorio de máscaras")
    parser.add_argument("--save-resolution", "-r", type=int, default=3840, 
                        help="Ancho en píxeles para guardar imágenes (default: 3840 = 4K)")
    parser.add_argument("--output-dir", "-o", default=None, 
                        help="Directorio de salida para screenshots (default: script_dir/figures)")
    parser.add_argument("--batch-save", "--save-all", dest="batch_save", action="store_true",
                        help="Recorrer todas las imágenes y guardar PNG sin abrir ventanas")
    
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
        print("No se encontraron imágenes.")
        return
    
    print(f"\nEncontradas {len(images)} imágenes")
    if args.batch_save:
        print("Modo batch: guardando PNG sin abrir ventanas.\n")
    else:
        print("Cierra cada ventana para pasar a la siguiente.\n")
    
    for i, name in enumerate(images):
        print(f"[{i+1}/{len(images)}] {name}")
        
        try:
            handler.load_pair(name, level=args.level)
            
            meta = handler.get_metadata()
            print(f"  Tamaño base: {meta.get('base_size')}")
            print(f"  Nivel cargado: {meta.get('current_level')}")
            
            # Mostrar información clínica si está disponible
            clinical = handler.get_clinical_info(name)
            if clinical:
                print(f"  Diagnóstico: {clinical.get('diagnosis', 'N/A')}")
                # Datos del paciente
                age = clinical.get('age', '')
                psa = clinical.get('psa', '')
                if age or psa:
                    patient_info = []
                    if age:
                        patient_info.append(f"Edad: {age}")
                    if psa:
                        patient_info.append(f"PSA: {psa}")
                    print(f"  {' | '.join(patient_info)}")
                # MRI
                mri = clinical.get('mri_findings', '')
                if mri:
                    print(f"  MRI: {mri}")
                # Clasificación
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
    
    print("\nVisualizacion completada.")


if __name__ == "__main__":
    main()
