import os
import io
from typing import Optional, List
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import segyio

app = FastAPI(title="SEG-Y File API", description="API for reading SEG-Y files")

SEGY_PATH = os.getenv("SEGY_PATH", "data/your.sgy")

def get_segy_file():
    """Get the SEG-Y file handle"""
    if not os.path.exists(SEGY_PATH):
        raise HTTPException(status_code=404, detail=f"SEG-Y file not found at {SEGY_PATH}")
    return segyio.open(SEGY_PATH, "r")

@app.get("/meta")
def get_metadata():
    """Get SEG-Y file metadata"""
    with get_segy_file() as f:
        return {
            "traces": len(f.trace),
            "samples_per_trace": len(f.samples),
            "sample_interval_us": f.bin[segyio.BinField.Interval]
        }

@app.get("/section")
def get_section(
    from_trace: int = Query(0, alias="from", ge=0),
    count: int = Query(200, ge=1),
    clip: float = Query(0.98, ge=0.0, le=1.0),
    agc: int = Query(200, ge=1)
):
    """Generate section slice as PNG grayscale image"""
    with get_segy_file() as f:
        total_traces = len(f.trace)
        
        if from_trace >= total_traces:
            raise HTTPException(status_code=400, detail="from_trace exceeds total traces")
        
        end_trace = min(from_trace + count, total_traces)
        actual_count = end_trace - from_trace
        
        # Read trace data
        data = np.zeros((len(f.samples), actual_count))
        for i, trace_idx in enumerate(range(from_trace, end_trace)):
            data[:, i] = f.trace[trace_idx]
        
        # Apply AGC (Automatic Gain Control)
        data = apply_agc(data, window_length=agc)
        
        # Apply clipping for contrast
        data = apply_clipping(data, clip)
        
        # Convert to 8-bit grayscale
        data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        
        # Create PIL Image
        img = Image.fromarray(data_normalized, mode='L')
        
        # Convert to PNG bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(img_buffer.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=section_{from_trace}_{actual_count}.png"}
        )

@app.get("/trace/{idx}")
def get_trace(idx: int):
    """Get single trace amplitude data as JSON"""
    with get_segy_file() as f:
        if idx < 0 or idx >= len(f.trace):
            raise HTTPException(status_code=400, detail=f"Trace index {idx} out of range (0-{len(f.trace)-1})")
        
        trace_data = f.trace[idx].tolist()
        return {
            "trace_index": idx,
            "amplitudes": trace_data,
            "sample_count": len(trace_data)
        }

def apply_agc(data: np.ndarray, window_length: int) -> np.ndarray:
    """Apply Automatic Gain Control"""
    result = data.copy()
    n_samples, n_traces = data.shape
    half_window = window_length // 2
    
    for i in range(n_samples):
        start_idx = max(0, i - half_window)
        end_idx = min(n_samples, i + half_window + 1)
        
        for j in range(n_traces):
            window_data = data[start_idx:end_idx, j]
            rms = np.sqrt(np.mean(window_data**2))
            if rms > 0:
                result[i, j] = data[i, j] / rms
    
    return result

def apply_clipping(data: np.ndarray, clip: float) -> np.ndarray:
    """Apply quantile clipping for contrast enhancement"""
    lower_percentile = (1 - clip) / 2 * 100
    upper_percentile = (1 - (1 - clip) / 2) * 100
    
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    
    return np.clip(data, lower_bound, upper_bound)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)