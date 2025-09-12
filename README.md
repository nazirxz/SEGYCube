# SEG-Y FastAPI Application

FastAPI application for reading and processing SEG-Y seismic data files.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your SEG-Y file:
   - Place your SEG-Y file in the `data/` directory as `your.sgy`
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

## API Endpoints

### GET /meta
Returns SEG-Y file metadata including:
- `traces`: Number of traces in the file
- `samples_per_trace`: Number of samples per trace
- `sample_interval_us`: Sample interval in microseconds

### GET /section
Generates a seismic section as PNG grayscale image.

Query parameters:
- `from`: Starting trace index (default: 0)
- `count`: Number of traces to include (default: 200)
- `clip`: Quantile for contrast clipping (default: 0.98)
- `agc`: Window length for Automatic Gain Control (default: 200)

### GET /trace/{idx}
Returns amplitude data for a single trace as JSON array.

Parameters:
- `idx`: Trace index (0-based)

Returns:
- `trace_index`: The requested trace index
- `amplitudes`: Array of amplitude values
- `sample_count`: Number of samples in the trace

## Environment Variables

- `SEGY_PATH`: Path to the SEG-Y file (default: "data/your.sgy")