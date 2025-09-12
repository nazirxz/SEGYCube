# SEG-Y 3D Cube API Documentation

## Overview
API ini menyediakan akses ke data seismik SEG-Y dengan kemampuan visualisasi dan analisis 3D yang lengkap. Semua endpoint mendukung caching untuk performa optimal dan CORS untuk integrasi web.

**Base URL**: `http://localhost:8100`

---

## 🆕 What's New - 3D Cube Implementation

### ✅ Major Improvements:
1. **True 3D Volume**: Data sekarang diorganisir sebagai proper 3D volume (inline × crossline × samples)
2. **Coordinate Awareness**: Mendukung inline/crossline coordinate system yang sesungguhnya  
3. **Memory Efficient**: Chunked loading dan parameter limit untuk memory management
4. **Binary Format**: Efficient binary data transfer untuk dataset besar
5. **Enhanced Mesh**: Marching cubes bekerja pada true 3D volume, bukan duplicate 2D
6. **Better Caching**: Improved cache system dengan proper key generation

### 📊 Current Survey Information:
- **Survey Type**: 3D Seismic Survey
- **Dimensions**: 138 inlines × 201 crosslines = 27,738 traces
- **Samples per Trace**: 451 samples
- **Sample Interval**: 10,000 μs (10 ms)
- **Coordinate Ranges**:
  - Inlines: 1228529611 to 1228562496
  - Crosslines: 1202458925 to 1202842905

---

## Error Responses

All endpoints may return these error responses:

### 404 Not Found
```json
{
  "detail": "SEG-Y file not found at data/depth_mig08_structural.bri_3D.segy"
}
```

### 400 Bad Request
```json
{
  "detail": "Trace index 2000 out of range (0-1199)"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["query", "clip"],
      "msg": "ensure this value is less than or equal to 1.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

## Endpoints

### GET /meta

**Description:** Get basic metadata about the SEG-Y file.

**Parameters:** None

**Response Model:**
```typescript
{
  traces: number;              // Total number of traces
  samples_per_trace: number;   // Samples per trace
  sample_interval_us: number;  // Sample interval in microseconds
}
```

**Example Response:**
```json
{
  "traces": 1200,
  "samples_per_trace": 2000,
  "sample_interval_us": 4000
}
```

**Usage:**
```bash
curl http://localhost:8100/meta
```

---

### GET /section

**Description:** Generate a 2D seismic section image as PNG with processing options.

**Query Parameters:**
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|--------|-------------|
| `from` | integer | 0 | ≥ 0 | Starting trace index |
| `count` | integer | 200 | ≥ 1 | Number of traces to include |
| `clip` | float | 0.98 | 0.0 - 1.0 | Quantile for contrast clipping |
| `agc` | integer | 200 | ≥ 1 | AGC window length in samples |

**Response:** Binary PNG image data

**Headers:**
```
Content-Type: image/png
Content-Disposition: inline; filename=section_{from}_{count}.png
```

**Processing Details:**
1. **Automatic Gain Control (AGC):** Normalizes amplitudes using a sliding window
2. **Quantile Clipping:** Enhances contrast by clipping extreme values
3. **8-bit Conversion:** Converts to grayscale values (0-255)

**Usage Examples:**
```bash
# Basic section (first 200 traces)
curl "http://localhost:8100/section" -o section.png

# Custom range with processing
curl "http://localhost:8100/section?from=100&count=300&clip=0.95&agc=150" -o section.png

# High contrast section
curl "http://localhost:8100/section?clip=0.9&agc=100" -o high_contrast.png
```

**Image Dimensions:**
- Width: `count` parameter (number of traces)
- Height: `samples_per_trace` from metadata

---

### GET /trace/{idx}

**Description:** Get amplitude data for a single trace.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `idx` | integer | Trace index (0-based) |

**Response Model:**
```typescript
{
  trace_index: number;    // Requested trace index
  amplitudes: number[];   // Array of amplitude values
  sample_count: number;   // Number of samples
}
```

**Example Response:**
```json
{
  "trace_index": 150,
  "amplitudes": [
    0.1234, -0.5678, 0.9012, -0.3456, 0.7890,
    -0.2345, 0.6789, -0.1023, 0.4567, -0.8901
  ],
  "sample_count": 2000
}
```

**Usage Examples:**
```bash
# Get first trace
curl http://localhost:8100/trace/0

# Get specific trace
curl http://localhost:8100/trace/150

# Save trace data to file
curl http://localhost:8100/trace/150 -o trace_150.json
```

**Data Format:**
- Amplitudes are 32-bit floating point values
- Array length equals `samples_per_trace` from metadata
- Values represent seismic amplitude at each time/depth sample

---

## Interactive Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI:** `http://localhost:8100/docs`
- **ReDoc:** `http://localhost:8100/redoc`
- **OpenAPI Schema:** `http://localhost:8100/openapi.json`

## Performance Considerations

- **Memory Usage:** Section endpoint loads multiple traces into memory
- **File Access:** Uses memory-mapped file access for efficiency
- **Concurrent Requests:** FastAPI handles multiple requests concurrently
- **Image Generation:** PNG compression is CPU-intensive for large sections

## Typical Workflows

### 1. Data Exploration
```bash
# Get file info
curl http://localhost:8100/meta

# Generate overview section
curl "http://localhost:8100/section?count=500" -o overview.png

# Examine specific traces
curl http://localhost:8100/trace/0 | jq '.amplitudes[:10]'
```

### 2. Quality Control
```bash
# Generate multiple sections for QC
curl "http://localhost:8100/section?from=0&count=200&agc=100" -o qc_section1.png
curl "http://localhost:8100/section?from=200&count=200&agc=100" -o qc_section2.png
curl "http://localhost:8100/section?from=400&count=200&agc=100" -o qc_section3.png
```

### 3. Detailed Analysis
```bash
# High-resolution section with custom processing
curl "http://localhost:8100/section?from=100&count=50&clip=0.99&agc=50" -o detailed.png

# Export trace data for analysis
for i in {100..150}; do
  curl "http://localhost:8100/trace/$i" -o "trace_$i.json"
done
```

---

## 3D Visualization Endpoints

### GET /volume

**Description:** Generate 3D volume data as depth slices for volume rendering.

**Query Parameters:**
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|--------|-------------|
| `depth_slices` | integer | 50 | 10-200 | Number of horizontal slices |
| `agc` | integer | 200 | ≥ 1 | AGC window length |
| `clip` | float | 0.98 | 0.0-1.0 | Quantile clipping |

**Response Model:**
```typescript
{
  dimensions: {
    traces: number;
    samples: number;
    depth_slices: number;
    grid_size: number;
  };
  data_info: {
    sample_interval_us: number;
    trace_spacing: number;
    processing: string;
    depth_indices: number[];
  };
  volume_slices: string[];  // Base64 encoded PNG images
}
```

**Usage:**
```bash
curl "http://localhost:8100/volume?depth_slices=64&agc=150" | jq '.dimensions'
```

---

## 🎲 3D Volume Endpoints

### GET `/cube` ⭐ **NEW**
Mendapatkan data 3D cube seismik yang terorganisir dengan koordinat inline/crossline.

**Parameters:**
- `subsample` (int): Faktor subsampling untuk samples (1-10, default: 4)
- `max_inlines` (int): Maksimum inline untuk diproses (10-138, default: 50)
- `max_crosslines` (int): Maksimum crossline untuk diproses (10-201, default: 50)
- `agc` (int): AGC window length (default: 200)
- `clip` (float): Quantile clipping (default: 0.98)
- `format` (str): Output format `json` atau `binary` (default: json)

**JSON Response:**
```json
{
    "dimensions": {
        "n_inlines": 50,
        "n_crosslines": 50,
        "n_samples": 113
    },
    "coordinate_info": {
        "inlines": {
            "start": 1228529611,
            "end": 1228541374,
            "count": 50
        },
        "crosslines": {
            "start": 1202458925,
            "end": 1202842905,
            "count": 50
        },
        "samples": {
            "start": 0,
            "end": 448,
            "count": 113,
            "subsample_factor": 4
        }
    },
    "processing": {
        "agc_window": 200,
        "clip_factor": 0.98,
        "data_range": {
            "min": -1.234,
            "max": 1.567
        }
    },
    "data_shape": [50, 50, 113],
    "data_type": "uint8",
    "survey_type": "3D",
    "access_info": {
        "binary_endpoint": "/cube?format=binary&...",
        "usage": "Use binary format for large datasets"
    }
}
```

**Binary Response:**
- Content-Type: `application/octet-stream`
- Headers:
  - `X-Cube-Inlines`: Number of inlines
  - `X-Cube-Crosslines`: Number of crosslines  
  - `X-Cube-Samples`: Number of samples
  - `X-Data-Type`: Data type (uint8)
  - `X-Inline-Start/End`: Inline coordinate range
  - `X-Crossline-Start/End`: Crossline coordinate range

**Examples:**
```bash
# Get 3D cube metadata
curl "http://localhost:8100/cube?max_inlines=30&max_crosslines=30"

# Download binary cube data
curl "http://localhost:8100/cube?format=binary&max_inlines=20&max_crosslines=20" \
  -o seismic_cube.raw

# High resolution cube (memory intensive)
curl "http://localhost:8100/cube?max_inlines=100&max_crosslines=100&subsample=2"
```

---

## 🏔️ 3D Mesh Generation

### GET `/mesh` ⭐ **ENHANCED**
Generate 3D mesh menggunakan marching cubes algorithm dari true 3D volume.

**Parameters:**
- `threshold` (float): Isosurface threshold 0.0-1.0 (default: 0.5)
- `subsample` (int): Subsampling factor (1-10, default: 4)
- `max_inlines` (int): Maximum inlines (10-138, default: 50)
- `max_crosslines` (int): Maximum crosslines (10-201, default: 50)
- `agc` (int): AGC window length (default: 200)
- `clip` (float): Quantile clipping (default: 0.98)

**Response:**
```json
{
    "vertices": [[x, y, z], [x, y, z], ...],
    "faces": [[v1, v2, v3], [v1, v2, v3], ...],
    "normals": [[nx, ny, nz], [nx, ny, nz], ...],
    "metadata": {
        "vertex_count": 15420,
        "face_count": 30840,
        "threshold": 0.5,
        "volume_dimensions": {
            "inlines": 50,
            "crosslines": 50,
            "samples": 113
        },
        "coordinate_ranges": {
            "inlines": [1228529611, 1228541374],
            "crosslines": [1202458925, 1202842905],
            "samples": [0, 448]
        },
        "survey_type": "3D",
        "cached": false
    }
}
```

**Examples:**
```bash
# Generate basic 3D mesh
curl "http://localhost:8100/mesh?threshold=0.6"

# High-detail mesh (slower)
curl "http://localhost:8100/mesh?max_inlines=80&max_crosslines=80&subsample=2&threshold=0.4"

# Fast mesh generation
curl "http://localhost:8100/mesh?max_inlines=20&max_crosslines=20&subsample=8"
```

---

### GET /volume/texture

**Description:** Generate raw 3D texture data for volume rendering.

**Query Parameters:**
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|--------|-------------|
| `resolution` | integer | 128 | 64-512 | Texture resolution per slice |
| `depth_slices` | integer | 64 | 16-128 | Number of depth slices |
| `agc` | integer | 200 | ≥ 1 | AGC window length |
| `clip` | float | 0.98 | 0.0-1.0 | Quantile clipping |

**Response:** Raw binary data (application/octet-stream)

**Headers:**
```
X-Texture-Width: 128
X-Texture-Height: 128 
X-Texture-Depth: 64
X-Texture-Format: R8
Content-Length: 1048576
```

**Usage:**
```bash
# Download 128x128x64 volume texture
curl "http://localhost:8100/volume/texture?resolution=128&depth_slices=64" -o volume.raw

# High resolution texture (larger file)
curl "http://localhost:8100/volume/texture?resolution=256&depth_slices=128" -o volume_hires.raw
```

**Data Format:**
- Single channel (grayscale) 8-bit unsigned integers
- Data layout: [depth][height][width]
- Total size: resolution × resolution × depth_slices bytes

---

## 🚀 Usage Examples & Integration

### Web Visualization Integration
```javascript
// Get 3D cube metadata
const response = await fetch('/cube?max_inlines=50&max_crosslines=50');
const metadata = await response.json();

// Download binary cube data
const binaryResponse = await fetch('/cube?format=binary&max_inlines=50&max_crosslines=50');
const buffer = await binaryResponse.arrayBuffer();
const cubeData = new Uint8Array(buffer);

// Use in Three.js or WebGL
const texture = new THREE.DataTexture3D(
    cubeData, 
    metadata.data_shape[0], 
    metadata.data_shape[1], 
    metadata.data_shape[2]
);
```

### Python Analysis
```python
import requests
import numpy as np

# Get cube data
response = requests.get('http://localhost:8100/cube', params={
    'format': 'binary',
    'max_inlines': 50,
    'max_crosslines': 50,
    'subsample': 4
})

# Parse binary data
cube_data = np.frombuffer(response.content, dtype=np.uint8)
cube_data = cube_data.reshape(50, 50, 113)  # inlines x crosslines x samples

# Generate mesh
mesh_response = requests.get('http://localhost:8100/mesh', params={
    'threshold': 0.6,
    'max_inlines': 50,
    'max_crosslines': 50
})
mesh = mesh_response.json()
```

### Performance Tips
1. **Untuk eksplorasi cepat**: Gunakan `subsample=8`, `max_inlines=20`, `max_crosslines=20`
2. **Untuk visualisasi detail**: Gunakan `subsample=2`, `max_inlines=80`, `max_crosslines=80`
3. **Untuk produksi**: Gunakan binary format dan caching
4. **Memory usage**: ~50x50x113 = 282KB per cube, ~100x100x225 = 2.25MB per cube

### Cache Management
```bash
# Clear all cached data
curl -X DELETE http://localhost:8100/cache

# Monitor cache usage (check disk space in cache/ directory)
```

---

## 🔄 Migration Guide

### From Legacy Endpoints:
- `/volume` endpoint masih tersedia untuk backward compatibility
- `/mesh` endpoint enhanced dengan 3D volume support
- New `/cube` endpoint untuk direct 3D cube access

### Parameter Changes:
- Mesh endpoint sekarang memiliki `max_inlines` dan `max_crosslines` parameters
- Binary format tersedia untuk efficient data transfer
- Coordinate information tersedia dalam response metadata

---

## 📝 Error Handling

Semua endpoint mengembalikan HTTP error codes yang sesuai:
- `400`: Parameter tidak valid atau data insufficient
- `404`: File tidak ditemukan atau operation tidak ada  
- `500`: Server error atau processing failure

Error responses:
```json
{
    "detail": "Volume dimensions (1, 50, 113) too small for mesh generation. Need at least 2x2x2."
}
```

---

## 🎯 Future Enhancements

Possible future improvements:
- Real-time streaming untuk large datasets
- WebSocket support untuk progress monitoring
- Additional file formats (SEGD, etc.)
- Advanced visualization filters
- Multi-threaded processing options