# SEG-Y API Documentation

## Overview
This API provides endpoints to read and process SEG-Y seismic data files. It supports metadata extraction, seismic section visualization, and individual trace data access.

## Base URL
```
http://localhost:8100
```

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

### GET /mesh

**Description:** Generate 3D mesh using marching cubes for isosurface extraction.

**Query Parameters:**
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|--------|-------------|
| `threshold` | float | 0.5 | 0.0-1.0 | Isosurface threshold |
| `subsample` | integer | 4 | 1-10 | Subsampling factor |
| `agc` | integer | 200 | ≥ 1 | AGC window length |
| `clip` | float | 0.98 | 0.0-1.0 | Quantile clipping |

**Response Model:**
```typescript
{
  vertices: number[][];     // [x, y, z] coordinates
  faces: number[][];        // Triangle indices
  normals: number[][];      // Surface normals
  metadata: {
    vertex_count: number;
    face_count: number;
    threshold: number;
    subsample_factor: number;
    original_dimensions: object;
    processed_dimensions: object;
    data_range: object;
  };
}
```

**Usage:**
```bash
# Generate mesh at 50% amplitude threshold
curl "http://localhost:8100/mesh?threshold=0.5&subsample=2" -o mesh.json

# High detail mesh (slower)
curl "http://localhost:8100/mesh?threshold=0.3&subsample=1" -o detailed_mesh.json
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