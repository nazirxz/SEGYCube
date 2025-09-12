import os
import io
import json
import base64
import pickle
import hashlib
from functools import lru_cache
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import segyio
from scipy import ndimage
from skimage import measure

app = FastAPI(title="SEG-Y File API", description="API for reading SEG-Y files")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:5000",  # Common development port
        "http://localhost:8080",  # Vue/other frameworks
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

SEGY_PATH = os.getenv("SEGY_PATH", "data/depth_mig08_structural.bri_3D.segy")
CACHE_DIR = "cache"

# Global cache for loaded data
_file_cache = {}
_metadata_cache = {}

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Response Models
class MetadataResponse(BaseModel):
    traces: int
    samples_per_trace: int
    sample_interval_us: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "traces": 1200,
                "samples_per_trace": 2000,
                "sample_interval_us": 4000
            }
        }

class TraceResponse(BaseModel):
    trace_index: int
    amplitudes: List[float]
    sample_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "trace_index": 150,
                "amplitudes": [0.1234, -0.5678, 0.9012, -0.3456],
                "sample_count": 2000
            }
        }

class VolumeResponse(BaseModel):
    dimensions: Dict[str, int]
    data_info: Dict[str, Any]
    volume_slices: List[str]  # Base64 encoded images
    
    class Config:
        json_schema_extra = {
            "example": {
                "dimensions": {
                    "traces": 1200,
                    "samples": 2000,
                    "depth_slices": 100
                },
                "data_info": {
                    "sample_interval_us": 4000,
                    "trace_spacing": 25.0,
                    "processing": "agc_applied"
                },
                "volume_slices": ["iVBORw0KGgoAAAANSUhEUgAA..."]
            }
        }

class MeshResponse(BaseModel):
    vertices: List[List[float]]
    faces: List[List[int]]
    normals: List[List[float]]
    metadata: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "faces": [[0, 1, 2], [1, 2, 3]],
                "normals": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                "metadata": {
                    "vertex_count": 1234,
                    "face_count": 2468,
                    "threshold": 0.5
                }
            }
        }

class ProgressResponse(BaseModel):
    status: str
    progress: float
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "processing", 
                "progress": 0.65,
                "message": "Loading trace data: 650/1000 traces"
            }
        }

# Progress tracking
_progress_cache = {}

def update_progress(operation_id: str, progress: float, message: str):
    """Update progress for long-running operations"""
    _progress_cache[operation_id] = {
        "status": "processing" if progress < 1.0 else "completed",
        "progress": progress,
        "message": message
    }

def get_cache_key(file_path: str, operation: str, **params) -> str:
    """Generate cache key for operations"""
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    content = f"{file_path}_{operation}_{param_str}"
    return hashlib.md5(content.encode()).hexdigest()

def get_cached_data(cache_key: str):
    """Get data from disk cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return None

def save_cached_data(cache_key: str, data):
    """Save data to disk cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass

@lru_cache(maxsize=1)
def get_segy_metadata(file_path: str):
    """Get SEG-Y metadata with caching"""
    if file_path in _metadata_cache:
        return _metadata_cache[file_path]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"SEG-Y file not found at {file_path}")
    
    with segyio.open(file_path, "r") as f:
        metadata = {
            "traces": len(f.trace),
            "samples_per_trace": len(f.samples),
            "sample_interval_us": f.bin[segyio.BinField.Interval]
        }
    
    _metadata_cache[file_path] = metadata
    return metadata

def get_segy_file():
    """Get the SEG-Y file handle"""
    if not os.path.exists(SEGY_PATH):
        raise HTTPException(status_code=404, detail=f"SEG-Y file not found at {SEGY_PATH}")
    return segyio.open(SEGY_PATH, "r")

def load_trace_chunk(file_path: str, start_trace: int, end_trace: int, subsample: int = 1):
    """Load a chunk of traces with optional subsampling"""
    cache_key = get_cache_key(file_path, "chunk", start=start_trace, end=end_trace, subsample=subsample)
    
    # Try cache first
    cached_data = get_cached_data(cache_key)
    if cached_data is not None:
        return cached_data
    
    with segyio.open(file_path, "r") as f:
        trace_count = min(end_trace - start_trace, len(f.trace) - start_trace)
        samples_per_trace = len(f.samples)
        
        # Subsample indices
        sample_indices = np.arange(0, samples_per_trace, subsample)
        trace_indices = np.arange(start_trace, start_trace + trace_count)
        
        # Load data
        data = np.zeros((len(trace_indices), len(sample_indices)))
        for i, trace_idx in enumerate(trace_indices):
            full_trace = f.trace[trace_idx]
            data[i, :] = full_trace[sample_indices]
    
    # Cache the result
    save_cached_data(cache_key, data)
    return data

@app.get("/meta", response_model=MetadataResponse)
def get_metadata():
    """
    Get SEG-Y file metadata (cached for performance)
    
    Returns basic information about the SEG-Y file including:
    - Number of traces in the file
    - Number of samples per trace
    - Sample interval in microseconds
    """
    return get_segy_metadata(SEGY_PATH)

@app.get("/section", 
         responses={
             200: {
                 "content": {"image/png": {}},
                 "description": "PNG grayscale image of seismic section"
             }
         })
def get_section(
    from_trace: int = Query(0, alias="from", ge=0, description="Starting trace index"),
    count: int = Query(200, ge=1, description="Number of traces to include"),
    clip: float = Query(0.98, ge=0.0, le=1.0, description="Quantile for contrast clipping (0.0-1.0)"),
    agc: int = Query(200, ge=1, description="Window length for Automatic Gain Control")
):
    """
    Generate seismic section slice as PNG grayscale image
    
    Creates a 2D seismic section image from the specified trace range with:
    - Automatic Gain Control (AGC) for amplitude normalization
    - Quantile clipping for contrast enhancement
    - Grayscale PNG output format
    
    The resulting image dimensions are:
    - Width: number of traces requested (count parameter)
    - Height: number of samples per trace
    """
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

@app.get("/trace/{idx}", response_model=TraceResponse)
def get_trace(idx: int):
    """
    Get single trace amplitude data as JSON
    
    Returns the complete amplitude array for a specific trace index.
    Useful for detailed analysis of individual seismic traces.
    
    Args:
        idx: Trace index (0-based, must be within valid range)
        
    Returns:
        JSON object containing trace index, amplitude values, and sample count
    """
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

@app.get("/volume", response_model=VolumeResponse)
def get_volume(
    depth_slices: int = Query(50, ge=10, le=200, description="Number of depth slices to generate"),
    agc: int = Query(200, ge=1, description="Window length for Automatic Gain Control"),
    clip: float = Query(0.98, ge=0.0, le=1.0, description="Quantile for contrast clipping")
):
    """
    Generate 3D volume data as a series of depth slices for volume rendering
    
    Creates multiple horizontal slices through the seismic volume at different depths.
    Each slice is returned as a base64-encoded PNG image that can be reconstructed
    into a 3D texture on the client side.
    
    Args:
        depth_slices: Number of horizontal slices to generate
        agc: AGC window length for amplitude normalization
        clip: Quantile clipping for contrast enhancement
        
    Returns:
        JSON object containing volume dimensions, metadata, and base64-encoded slice images
    """
    with get_segy_file() as f:
        total_traces = len(f.trace)
        samples_per_trace = len(f.samples)
        
        # Load all trace data into 3D array (traces x samples)
        data = np.zeros((total_traces, samples_per_trace))
        for i in range(total_traces):
            data[i, :] = f.trace[i]
        
        # Apply processing
        data = apply_agc(data.T, window_length=agc).T  # Transpose for AGC, then back
        data = apply_clipping(data, clip)
        
        # Generate depth slices
        slice_images = []
        slice_indices = np.linspace(0, samples_per_trace-1, depth_slices, dtype=int)
        
        for depth_idx in slice_indices:
            # Extract horizontal slice at this depth
            slice_data = data[:, depth_idx].reshape(1, -1)  # 1 x traces
            
            # Normalize to 0-255
            slice_normalized = ((slice_data - slice_data.min()) / 
                              (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            
            # Create square-ish image (approximate grid arrangement of traces)
            grid_size = int(np.ceil(np.sqrt(total_traces)))
            padded_data = np.zeros((grid_size * grid_size,))
            padded_data[:total_traces] = slice_normalized.flatten()
            slice_image = padded_data.reshape(grid_size, grid_size).astype(np.uint8)
            
            # Convert to PNG
            img = Image.fromarray(slice_image, mode='L')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            slice_images.append(img_b64)
        
        return VolumeResponse(
            dimensions={
                "traces": total_traces,
                "samples": samples_per_trace,
                "depth_slices": depth_slices,
                "grid_size": grid_size
            },
            data_info={
                "sample_interval_us": f.bin[segyio.BinField.Interval],
                "trace_spacing": 25.0,  # Assumed
                "processing": f"agc_window_{agc}_clip_{clip}",
                "depth_indices": slice_indices.tolist()
            },
            volume_slices=slice_images
        )

@app.get("/mesh", response_model=MeshResponse)
def get_mesh(
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Isosurface threshold (0-1)"),
    subsample: int = Query(4, ge=1, le=10, description="Subsampling factor to reduce data size"),
    agc: int = Query(200, ge=1, description="AGC window length"),
    clip: float = Query(0.98, ge=0.0, le=1.0, description="Quantile clipping")
):
    """
    Generate 3D mesh using marching cubes algorithm for isosurface extraction (cached)
    
    Creates a 3D mesh representation of seismic horizons at a specified amplitude threshold.
    Uses the marching cubes algorithm to extract isosurfaces from the volume data.
    Results are cached to disk for faster subsequent requests.
    
    Args:
        threshold: Amplitude threshold for isosurface (normalized 0-1)
        subsample: Subsampling factor to reduce computation time
        agc: AGC window length for preprocessing
        clip: Quantile clipping for preprocessing
        
    Returns:
        JSON object containing vertices, faces, normals and metadata for 3D mesh
    """
    # Check cache first
    cache_key = get_cache_key(SEGY_PATH, "mesh", 
                             threshold=threshold, subsample=subsample, agc=agc, clip=clip)
    cached_result = get_cached_data(cache_key)
    if cached_result is not None:
        return cached_result
    
    try:
        metadata = get_segy_metadata(SEGY_PATH)
        total_traces = metadata["traces"]
        samples_per_trace = metadata["samples_per_trace"]
        
        # Use chunked loading for better memory management
        max_traces = min(1000, total_traces)  # Limit to manageable size
        end_trace = min(max_traces, total_traces)
        
        # Load data with chunked approach
        data = load_trace_chunk(SEGY_PATH, 0, end_trace, subsample)
        
        # Apply processing
        data = apply_agc(data.T, window_length=max(1, agc//subsample)).T
        data = apply_clipping(data, clip)
        
        # Normalize to 0-1 range
        data_min, data_max = data.min(), data.max()
        if data_max == data_min:
            raise HTTPException(status_code=400, detail="Data has no variation - cannot generate mesh")
        
        data_normalized = (data - data_min) / (data_max - data_min)
        
        # Create 3D volume (traces x samples x depth)
        volume = data_normalized[:, :, np.newaxis]
        
        # Extract isosurface using marching cubes
        vertices, faces, normals, values = measure.marching_cubes(
            volume, threshold, spacing=(1.0, 1.0, 1.0)
        )
        
        # Scale vertices to real-world coordinates
        vertices[:, 0] *= subsample  # trace spacing
        vertices[:, 1] *= subsample  # sample spacing
        
        result = MeshResponse(
            vertices=vertices.tolist(),
            faces=faces.tolist(),
            normals=normals.tolist(),
            metadata={
                "vertex_count": len(vertices),
                "face_count": len(faces),
                "threshold": threshold,
                "subsample_factor": subsample,
                "original_dimensions": {
                    "traces": total_traces,
                    "samples": samples_per_trace
                },
                "processed_dimensions": {
                    "traces": len(data),
                    "samples": len(data[0]) if len(data) > 0 else 0
                },
                "data_range": {
                    "min": float(data_min),
                    "max": float(data_max)
                },
                "cached": False
            }
        )
        
        # Cache the result
        save_cached_data(cache_key, result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")

@app.get("/volume/texture")
async def get_volume_texture(
    resolution: int = Query(128, ge=64, le=512, description="Texture resolution per slice"),
    depth_slices: int = Query(64, ge=16, le=128, description="Number of depth slices"),
    agc: int = Query(200, ge=1, description="AGC window length"),
    clip: float = Query(0.98, ge=0.0, le=1.0, description="Quantile clipping")
):
    """
    Generate volume texture data for 3D volume rendering
    
    Creates a 3D texture by stacking 2D slices from different depths.
    Returns raw texture data that can be used for volume rendering in WebGL.
    
    Args:
        resolution: Resolution of each texture slice (resolution x resolution)
        depth_slices: Number of depth slices to stack
        agc: AGC window length
        clip: Quantile clipping value
        
    Returns:
        Raw binary data representing 3D texture (width x height x depth x channels)
    """
    with get_segy_file() as f:
        total_traces = len(f.trace)
        samples_per_trace = len(f.samples)
        
        # Load all data
        data = np.zeros((total_traces, samples_per_trace))
        for i in range(total_traces):
            data[i, :] = f.trace[i]
        
        # Apply processing
        data = apply_agc(data.T, window_length=agc).T
        data = apply_clipping(data, clip)
        
        # Normalize to 0-255
        data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        
        # Create 3D texture array
        texture_data = np.zeros((depth_slices, resolution, resolution), dtype=np.uint8)
        
        # Sample depth slices
        depth_indices = np.linspace(0, samples_per_trace-1, depth_slices, dtype=int)
        
        for z, depth_idx in enumerate(depth_indices):
            # Extract slice at this depth
            slice_data = data_normalized[:, depth_idx]
            
            # Interpolate to resolution x resolution grid
            if total_traces != resolution * resolution:
                # Create grid arrangement
                grid_size = int(np.ceil(np.sqrt(total_traces)))
                padded_data = np.zeros((grid_size * grid_size,))
                padded_data[:total_traces] = slice_data
                grid_data = padded_data.reshape(grid_size, grid_size)
                
                # Resize to target resolution
                from PIL import Image
                img = Image.fromarray(grid_data.astype(np.uint8), mode='L')
                img_resized = img.resize((resolution, resolution), Image.LANCZOS)
                texture_data[z] = np.array(img_resized)
            else:
                # Direct mapping if dimensions match
                texture_data[z] = slice_data.reshape(resolution, resolution)
        
        # Convert to bytes
        texture_bytes = texture_data.tobytes()
        
        # Return raw binary data with appropriate headers
        return StreamingResponse(
            io.BytesIO(texture_bytes),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=volume_{resolution}x{resolution}x{depth_slices}.raw",
                "X-Texture-Width": str(resolution),
                "X-Texture-Height": str(resolution),
                "X-Texture-Depth": str(depth_slices),
                "X-Texture-Format": "R8",
                "Content-Length": str(len(texture_bytes))
            }
        )

@app.get("/progress/{operation_id}", response_model=ProgressResponse)
def get_progress(operation_id: str):
    """
    Get progress of long-running operations
    
    Args:
        operation_id: Unique identifier for the operation
        
    Returns:
        Progress status, percentage, and current message
    """
    if operation_id not in _progress_cache:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    return _progress_cache[operation_id]

@app.delete("/cache")
def clear_cache():
    """
    Clear all cached data to free up disk space
    
    Returns:
        Status message about cache clearing
    """
    import shutil
    try:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR)
        
        # Clear memory caches
        _file_cache.clear()
        _metadata_cache.clear()
        _progress_cache.clear()
        
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100, reload=True)