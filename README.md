# SEG-Y FastAPI Application

FastAPI application for reading and processing SEG-Y seismic data files.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your SEG-Y file:
   - Place your SEG-Y file in the `data/` directory as `depth_mig08_structural.bri_3D.segy`
   - Or set the `SEGY_PATH` environment variable to point to your file

## Running the Application

```bash
uvicorn fastapi_app:app --reload
```

Or run directly:
```bash
python fastapi_app.py
```

The API will be available at `http://localhost:8000`

**CORS Configuration:**
The API is configured to allow cross-origin requests from:
- `http://localhost:3000` (React)
- `http://localhost:5000` (General development)
- `http://localhost:8080` (Vue/other frameworks)
- `http://127.0.0.1:*` (IPv4 localhost variants)

To test CORS functionality, open `cors_test.html` in your browser after starting the server.

## API Endpoints

### GET /meta
Returns SEG-Y file metadata including basic file information.

**Response Example:**
```json
{
  "traces": 1200,
  "samples_per_trace": 2000,
  "sample_interval_us": 4000
}
```

**Response Fields:**
- `traces`: Number of traces in the file
- `samples_per_trace`: Number of samples per trace
- `sample_interval_us`: Sample interval in microseconds

### GET /section
Generates a seismic section as PNG grayscale image with processing options.

**Query Parameters:**
- `from`: Starting trace index (default: 0)
- `count`: Number of traces to include (default: 200)  
- `clip`: Quantile for contrast clipping, range 0.0-1.0 (default: 0.98)
- `agc`: Window length for Automatic Gain Control (default: 200)

**Response:** PNG image file with headers:
```
Content-Type: image/png
Content-Disposition: inline; filename=section_{from}_{count}.png
```

**Example Request:**
```bash
curl "http://localhost:8000/section?from=100&count=300&clip=0.95&agc=150" -o section.png
```

### GET /trace/{idx}
Returns complete amplitude data for a single trace as JSON.

**Path Parameters:**
- `idx`: Trace index (0-based, must be within valid range)

**Response Example:**
```json
{
  "trace_index": 150,
  "amplitudes": [0.1234, -0.5678, 0.9012, -0.3456, ...],
  "sample_count": 2000
}
```

**Response Fields:**
- `trace_index`: The requested trace index
- `amplitudes`: Array of amplitude values (floating point)
- `sample_count`: Number of samples in the trace

**Example Request:**
```bash
curl http://localhost:8000/trace/150
```

## 3D Visualization Endpoints

### GET /volume
Generate 3D volume data as depth slices for volume rendering. Returns base64-encoded PNG slices that can be reconstructed into 3D textures.

**Example Request:**
```bash
curl "http://localhost:8000/volume?depth_slices=64&agc=150" | jq '.dimensions'
```

### GET /mesh
Generate 3D mesh using marching cubes algorithm for isosurface extraction. Perfect for creating 3D geological horizon surfaces.

**Example Request:**
```bash
curl "http://localhost:8000/mesh?threshold=0.5&subsample=2" -o seismic_mesh.json
```

### GET /volume/texture
Generate raw 3D texture data for advanced volume rendering in WebGL/Three.js applications.

**Example Request:**
```bash
curl "http://localhost:8000/volume/texture?resolution=128&depth_slices=64" -o volume_texture.raw
```

## Environment Variables

- `SEGY_PATH`: Path to the SEG-Y file (default: "data/depth_mig08_structural.bri_3D.segy")