/**
 * QuPath GeoJSON Cropped Exporter
 *
 * Script para exportar anotaciones vectoriales como GeoJSON con coordenadas
 * relativas al mismo recorte espacial que export_cropped.groovy.
 *
 * Las coordenadas en el GeoJSON son relativas al bounding box del recorte
 * (valores en el rango [0, 1] con suficiente precisión). Esto hace que los
 * polígonos sean independientes de la resolución de la imagen original y
 * puedan superponerse directamente sobre cualquier versión del recorte
 * simplemente multiplicando por el ancho y alto de la imagen en píxeles.
 * Se incluye metadata con crop_width/crop_height para referencia.
 *
 * Decisiones de diseño:
 *   - Todas las clases de anotación se exportan (incluido Artifact).
 *   - Las geometrías se recortan al bounding box del crop para evitar
 *     coordenadas negativas o fuera de rango.
 *   - Sin CRS explícito (coordenadas relativas [0, 1] independientes de resolución).
 *   - Archivos nombrados como {name}_annotations.geojson para coincidir
 *     visualmente con {name}.ome.tif.
 *   - Una sola versión sin pirámide de resolución (full-res, nivel 0).
 *
 * Requisitos: QuPath 0.6.x o superior
 *
 * Uso:
 *   1. Abrir proyecto en QuPath
 *   2. Ajustar OUTPUT_DIR y parámetros (los mismos que export_cropped.groovy)
 *   3. Ejecutar script (Ctrl+R)
 */

import qupath.lib.objects.PathAnnotationObject
import qupath.lib.common.GeneralTools
import qupath.lib.roi.GeometryTools
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.util.AffineTransformation
import com.google.gson.Gson
import com.google.gson.GsonBuilder

// =====================================================
// CONFIGURACIÓN
// =====================================================

// Directorio de salida (mismo que export_cropped.groovy)
def OUTPUT_DIR = "/media/abel/TOSHIBA EXT/export"

// Subdirectorio para GeoJSON
def GEOJSON_SUBDIR = "geojson"

// Margen a añadir al bounding box (0.1 = 10%) — mismo que export_cropped
def MARGIN_RATIO = 0.1

// Clase a ignorar para calcular el bounding box (mismo criterio)
def IGNORE_CLASS_NAME = "Artifact"

// Crear directorio de salida
def geojsonDir = buildFilePath(OUTPUT_DIR, GEOJSON_SUBDIR)
mkdirs(geojsonDir)

// =====================================================
// DEFINICIÓN DE CLASES
// =====================================================

def CLASS_NAMES = [
    "Tumor",                                          // ID: 1
    "Benign gland",                                   // ID: 2
    "Blood vessels",                                  // ID: 3
    "Fibromuscular bundles",                          // ID: 4
    "Abnormal secretions",                            // ID: 5
    "Contamination with another tissue",              // ID: 6
    "Prominent nucleolus",                            // ID: 7
    "Immune cells",                                   // ID: 8
    "Nerve",                                          // ID: 9
    "Artifact",                                       // ID: 10
    "Seminal vesicle",                                // ID: 11
    "Adipose tissue",                                 // ID: 12
    "Normal  secretions",                             // ID: 13
    "Stromal retraction spaces",                      // ID: 14
    "Muscle",                                         // ID: 15
    "Foreign body contamination",                     // ID: 16
    "High grade prostatic intraepithelial neoplasia (HGPIN)", // ID: 17
    "Calcifications",                                 // ID: 18
    "Intestinal glands and mucus",                    // ID: 19
    "Perineural invasion (PNI)",                      // ID: 20
    "Hemorrhage",                                    // ID: 21
    "Intraductal carcinoma",                          // ID: 22
    "Necrosis",                                       // ID: 23
    "Mitosis",                                        // ID: 24
    "Nerve ganglion",                                 // ID: 25
    "Atypical intraductal proliferation",             // ID: 26
    "Red blood cells"                                 // ID: 27
]

// =====================================================
// FUNCIONES AUXILIARES
// =====================================================

/**
 * Calcula el bounding box combinado de todas las anotaciones,
 * ignorando las de la clase especificada (típicamente "Artifact").
 *
 * Por qué ignorar una clase en el bbox sin excluirla del GeoJSON:
 *   Las anotaciones de Artifact suelen marcar bordes de escaneo o
 *   regiones dañadas, y no queremos que expandan el recorte
 *   innecesariamente. Pero sí las exportamos en el GeoJSON para
 *   que el consumidor pueda decidir si usarlas o filtrarlas.
 *
 * @return [x, y, width, height] o null si no hay anotaciones válidas
 */
def calculateAnnotationsBBox(annotations, ignoreClassName) {
    def validAnnotations = annotations.findAll { ann ->
        def pc = ann.getPathClass()
        pc == null || pc.getName() != ignoreClassName
    }

    if (validAnnotations.isEmpty()) {
        return null
    }

    def minX = Double.MAX_VALUE
    def minY = Double.MAX_VALUE
    def maxX = Double.MIN_VALUE
    def maxY = Double.MIN_VALUE

    validAnnotations.each { ann ->
        def roi = ann.getROI()
        def boundsX = roi.getBoundsX()
        def boundsY = roi.getBoundsY()
        def boundsW = roi.getBoundsWidth()
        def boundsH = roi.getBoundsHeight()

        minX = Math.min(minX, boundsX)
        minY = Math.min(minY, boundsY)
        maxX = Math.max(maxX, boundsX + boundsW)
        maxY = Math.max(maxY, boundsY + boundsH)
    }

    return [
        (int) minX,
        (int) minY,
        (int) (maxX - minX),
        (int) (maxY - minY)
    ]
}

/**
 * Añade margen al bounding box, respetando los límites de la imagen.
 */
def addMargin(bbox, imageWidth, imageHeight, marginRatio) {
    def (x, y, w, h) = bbox

    def marginX = (int) (w * marginRatio)
    def marginY = (int) (h * marginRatio)

    def newX = Math.max(0, x - marginX)
    def newY = Math.max(0, y - marginY)
    def newMaxX = Math.min(imageWidth, x + w + marginX)
    def newMaxY = Math.min(imageHeight, y + h + marginY)

    return [
        newX,
        newY,
        newMaxX - newX,
        newMaxY - newY
    ]
}

/**
 * Convierte un ROI de QuPath a geometría JTS, la traslada al espacio
 * del recorte y la recorta al bounding box del crop.
 *
 * Pipeline:
 *   1. ROI → JTS Geometry (via GeometryTools)
 *   2. Traslación: (x, y) → (x - cropX, y - cropY)
 *   3. Intersección con el rectángulo del crop [0,0,cropW,cropH]
 *
 * Las operaciones JTS se realizan en espacio de píxeles para preservar
 * la precisión numérica del clipping. La normalización a [0,1] se hace
 * después, en jtsToGeoJSONGeometry(), al serializar las coordenadas.
 *
 * Si la geometría queda completamente fuera del crop, retorna null.
 *
 * @return Geometría JTS recortada (en píxeles) o null
 */
def roiToCroppedGeometry(roi, cropX, cropY, cropW, cropH) {
    if (roi == null) return null

    // 1. ROI → JTS Geometry
    def geometry = new GeometryTools().roiToGeometry(roi)
    if (geometry == null || geometry.isEmpty()) {
        return null
    }

    // 2. Trasladar al espacio del recorte
    def translation = AffineTransformation.translationInstance(-cropX, -cropY)
    def translated = translation.transform(geometry)

	    // 3. Recortar al bounding box del crop en espacio de píxeles
	    //    (evitamos normalizar aquí para no degradar la precisión de JTS)
	    def gf = new GeometryFactory()
	    def cropEnvelope = gf.createPolygon([
	        new Coordinate(0, 0),
	        new Coordinate(cropW, 0),
	        new Coordinate(cropW, cropH),
	        new Coordinate(0, cropH),
	        new Coordinate(0, 0)
	    ] as Coordinate[])

    def clipped = translated.intersection(cropEnvelope)

    if (clipped == null || clipped.isEmpty()) {
        return null
    }

    return clipped
}

/**
 * Convierte una geometría JTS a estructura GeoJSON con coordenadas
 * relativas [0,1] (divide cada coordenada por cropW/cropH).
 *
 * Soporta: Polygon, MultiPolygon, Point, MultiPoint, LineString,
 *          MultiLineString, GeometryCollection.
 *
 * La normalización se aplica aquí, al serializar, para que las
 * operaciones JTS previas trabajen en espacio de píxeles con
 * máxima precisión numérica. No se aplica redondeo: se preservan
 * todos los puntos del contorno con precisión double nativa.
 */
def jtsToGeoJSONGeometry(Geometry geom, double cropW, double cropH) {
    if (geom == null || geom.isEmpty()) return null

    def type = geom.getGeometryType()
    def result = [type: type]

    switch (type) {
        case "Polygon":
            def rings = []
            rings << geom.getExteriorRing().getCoordinates().collect { [it.x / cropW, it.y / cropH] }
            for (int i = 0; i < geom.getNumInteriorRing(); i++) {
                rings << geom.getInteriorRingN(i).getCoordinates().collect { [it.x / cropW, it.y / cropH] }
            }
            result.coordinates = rings
            break

        case "MultiPolygon":
            def polys = []
            for (int i = 0; i < geom.getNumGeometries(); i++) {
                def poly = geom.getGeometryN(i)
                def rings = []
                rings << poly.getExteriorRing().getCoordinates().collect { [it.x / cropW, it.y / cropH] }
                for (int j = 0; j < poly.getNumInteriorRing(); j++) {
                    rings << poly.getInteriorRingN(j).getCoordinates().collect { [it.x / cropW, it.y / cropH] }
                }
                polys << rings
            }
            result.coordinates = polys
            break

        case "Point":
            def c = geom.getCoordinate()
            result.coordinates = [c.x / cropW, c.y / cropH]
            break

        case "MultiPoint":
            result.coordinates = geom.getCoordinates().collect { [it.x / cropW, it.y / cropH] }
            break

        case "LineString":
            result.coordinates = geom.getCoordinates().collect { [it.x / cropW, it.y / cropH] }
            break

        case "MultiLineString":
            def lines = []
            for (int i = 0; i < geom.getNumGeometries(); i++) {
                lines << geom.getGeometryN(i).getCoordinates().collect { [it.x / cropW, it.y / cropH] }
            }
            result.coordinates = lines
            break

        case "GeometryCollection":
            def geoms = []
            for (int i = 0; i < geom.getNumGeometries(); i++) {
                def child = jtsToGeoJSONGeometry(geom.getGeometryN(i), cropW, cropH)
                if (child != null) geoms << child
            }
            return [type: "GeometryCollection", geometries: geoms]

        default:
            return null
    }

    return result
}

/**
 * Devuelve el tipo de ROI como string legible (ej. "Polygon", "Rectangle", "Ellipse").
 * Útil para saber si una anotación es un bbox o una segmentación.
 */
def getRoiTypeName(roi) {
    if (roi == null) return "Unknown"
    def simpleName = roi.getClass().getSimpleName()
    return simpleName.replace("ROI", "")
}

// =====================================================
// VERIFICAR PROYECTO
// =====================================================

def project = getProject()
if (project == null) {
    print "ERROR: No hay proyecto abierto"
    return
}

print "=========================================="
print "QuPath GeoJSON Cropped Exporter"
print "=========================================="
print "Proyecto: ${project.getName()}"
print "Imagenes: ${project.getImageList().size()}"
print "Directorio de salida: ${OUTPUT_DIR}/${GEOJSON_SUBDIR}"
print "Margen: ${(MARGIN_RATIO * 100) as int}%"
print "Clase ignorada para bbox: ${IGNORE_CLASS_NAME}"
print "=========================================="
print ""

// =====================================================
// PROCESAR CADA IMAGEN DEL PROYECTO
// =====================================================

def processedCount = 0
def skippedCount = 0
def errorCount = 0
def totalFeatures = 0
def totalFeaturesSkipped = 0
def classCounts = [:]
def gson = new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create()

def startTime = System.currentTimeMillis()

project.getImageList().each { entry ->

    print "--------------------------------------------"
    print "Procesando: ${entry.getImageName()}"

    try {
        def imageData = entry.readImageData()
        def hierarchy = imageData.getHierarchy()
        def server = imageData.getServer()
        def imageName = GeneralTools.stripExtension(entry.getImageName())

        def origWidth = server.getWidth()
        def origHeight = server.getHeight()

        print "  Original: ${origWidth} x ${origHeight}"

        // Obtener TODAS las anotaciones (sin filtrar por clase)
        def annotations = hierarchy.getObjects(null, PathAnnotationObject)

        if (annotations.isEmpty()) {
            print "  Sin anotaciones - omitiendo"
            skippedCount++
            return
        }

        def annotationsWithROI = annotations.findAll { it.getROI() != null }
        print "  Total anotaciones: ${annotations.size()} (${annotationsWithROI.size()} con ROI)"

        // Calcular bounding box ignorando Artifact (mismo criterio que export_cropped)
        def bbox = calculateAnnotationsBBox(annotations, IGNORE_CLASS_NAME)

        if (bbox == null) {
            print "  Sin anotaciones validas para bbox (solo Artifact) - omitiendo"
            skippedCount++
            return
        }

        print "  Bbox anotaciones: x=${bbox[0]}, y=${bbox[1]}, w=${bbox[2]}, h=${bbox[3]}"

        // Añadir margen
        def bboxWithMargin = addMargin(bbox, origWidth, origHeight, MARGIN_RATIO)
        def (cropX, cropY, cropW, cropH) = bboxWithMargin

        print "  Bbox con margen: x=${cropX}, y=${cropY}, w=${cropW}, h=${cropH}"

        // =====================================================
        // CONSTRUIR FEATURES GEOJSON
        // =====================================================

        def features = []
        def featuresSkipped = 0

        annotationsWithROI.eachWithIndex { ann, idx ->
            def roi = ann.getROI()
            def pathClass = ann.getPathClass()

            // Resolver clasificación
            def className = pathClass?.getName() ?: "Unclassified"
            // Buscar ID en CLASS_NAMES (1-based); 0 si no está en la lista
            def classIdx = CLASS_NAMES.findIndexOf { it == className }
            def classId = (classIdx >= 0) ? classIdx + 1 : 0

            // Convertir, trasladar y recortar geometría
            def clippedGeom = roiToCroppedGeometry(roi, cropX, cropY, cropW, cropH)
            if (clippedGeom == null) {
                featuresSkipped++
                return
            }

            // Convertir geometría JTS a estructura GeoJSON
            def geometry = jtsToGeoJSONGeometry(clippedGeom, cropW, cropH)
            if (geometry == null) {
                featuresSkipped++
                return
            }

            def areaRelative = clippedGeom.getArea() / (cropW * cropH)

            def feature = [
                type       : "Feature",
                geometry   : geometry,
                properties : [
                    classification: [
                        name : className,
                        id   : classId
                    ],
                    roi_type      : getRoiTypeName(roi),
                    area_relative : areaRelative
                ]
            ]

            features << feature

            // Acumular contador por clase
            def key = "${classId}: ${className}"
            classCounts[key] = (classCounts[key] ?: 0) + 1
        }

        // =====================================================
        // ESCRIBIR ARCHIVO GEOJSON
        // =====================================================

        def geojsonPath = buildFilePath(geojsonDir, "${imageName}_annotations.geojson")

        def geojson = [
            type     : "FeatureCollection",
            features : features,
            metadata : [
                wsi_name          : "${imageName}.ome.tif".toString(),
                anonymous_code    : "${imageName}".toString(),
                coordinate_space  : "relative",
                width             : cropW,
                height            : cropH,
                total_annotations : features.size(),
                generated_by      : "export_geojson_cropped.groovy"
            ]
        ]

        def jsonString = gson.toJson(geojson)
        new File(geojsonPath).text = jsonString

        def fileSize = new File(geojsonPath).length()
        def fileSizeKB = Math.round(fileSize / 10.24) / 100.0
        print "  GeoJSON: ${new File(geojsonPath).getName()} (${fileSizeKB} KB, ${features.size()} features, ${featuresSkipped} skipped)"

        totalFeatures += features.size()
        totalFeaturesSkipped += featuresSkipped
        processedCount++

        // Liberar memoria
        imageData.getServer().close()
        System.gc()

    } catch (Exception e) {
        print "  ERROR: ${e.getMessage()}"
        e.printStackTrace()
        errorCount++
    }
}

// =====================================================
// RESUMEN FINAL
// =====================================================

def totalTime = (System.currentTimeMillis() - startTime) / 1000.0

print ""
print "=========================================="
print "PROCESO COMPLETADO"
print "=========================================="
print "Procesadas (con GeoJSON): ${processedCount}"
print "Omitidas (sin datos):    ${skippedCount}"
print "Errores:                  ${errorCount}"
print ""
print "Total features exportados: ${totalFeatures}"
print "Features descartados (fuera de crop): ${totalFeaturesSkipped}"
print ""
print "Instancias por clase:"
def sortedCounts = classCounts.sort { -it.value }
sortedCounts.each { cls, count ->
    print String.format("  %-60s %5d", cls, count)
}
print ""
print ""
print String.format("Tiempo total: %.1f segundos (%.1f minutos)", totalTime, totalTime / 60.0)
print "=========================================="
