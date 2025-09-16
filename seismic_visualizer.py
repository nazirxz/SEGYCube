# -*- coding: utf-8 -*-
"""
Skrip untuk memuat, memproses, dan memvisualisasikan data seismik 3D
dari file SEG-Y. Mendukung pemrosesan via CPU (NumPy) dan GPU (PyTorch/CUDA).
Visualisasi disajikan melalui server web lokal menggunakan Dash.
"""

# Step 1: Import Libraries
import os
import sys
import time
import segyio
import socket # Ditambahkan untuk mendeteksi IP lokal
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# --- Coba impor PyTorch ---
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- Coba impor Dash ---
try:
    import dash
    from dash import dcc, html
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


# --- Konfigurasi Awal ---
pio.renderers.default = "browser"

# --- PILIH MODE PEMROSESAN ---
# Atur ke True untuk mencoba menggunakan GPU.
USE_GPU = True


# --- Pengecekan Lingkungan ---
if TORCH_AVAILABLE and USE_GPU:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"✅ PyTorch terdeteksi. Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️ PyTorch terdeteksi, tetapi CUDA tidak tersedia. Menggunakan CPU.")
else:
    DEVICE = torch.device("cpu")
    if USE_GPU:
        print("❌ PyTorch tidak terinstal. Pemrosesan akan dilakukan di CPU menggunakan NumPy.")
    else:
        print("➡️ Pemrosesan diatur untuk berjalan di CPU menggunakan NumPy.")

print("\nLibraries imported successfully!")
print("👉 Jupyter is currently using this Python interpreter:")
print(sys.executable)


# Step 2: Configuration
SEGY_PATH = "data/depth_mig08_structural.bri_3D.segy"
MAX_INLINES = 32
MAX_CROSSLINES = 32
SUBSAMPLE = 8


# Step 3: Fungsi Pemuatan Data (Tetap menggunakan CPU)
def load_segy_volume(file_path, max_inlines, max_crosslines, subsample):
    """Memuat sebagian data dari file SEG-Y dan mengembalikan volume beserta koordinatnya."""
    if not os.path.exists(file_path):
        print(f"Error: SEG-Y file not found at {file_path}. Please check the path.")
        return None, None, None, None
    print(f"Loading 3D volume from: {file_path}...")
    try:
        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
            crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            unique_inlines = sorted(list(set(inlines)))
            unique_crosslines = sorted(list(set(crosslines)))
            inlines_to_load = unique_inlines[:max_inlines]
            crosslines_to_load = unique_crosslines[:max_crosslines]

            # Dapatkan interval sampel untuk sumbu Z (waktu/kedalaman)
            sample_interval_us = f.bin[segyio.BinField.Interval]
            sample_interval_ms = sample_interval_us / 1000.0

            samples_per_trace = len(f.samples)
            sample_indices = np.arange(0, samples_per_trace, subsample)
            
            # Hitung koordinat Z (waktu dalam ms)
            z_coords = sample_indices * sample_interval_ms

            volume = np.zeros((len(inlines_to_load), len(crosslines_to_load), len(sample_indices)), dtype=np.float32)
            inline_map = {inline: idx for idx, inline in enumerate(inlines_to_load)}
            crossline_map = {crossline: idx for idx, crossline in enumerate(crosslines_to_load)}
            for trace_idx in range(len(f.trace)):
                header = f.header[trace_idx]
                inline = header[segyio.TraceField.INLINE_3D]
                crossline = header[segyio.TraceField.CROSSLINE_3D]
                if inline in inline_map and crossline in crossline_map:
                    i_idx = inline_map[inline]
                    c_idx = crossline_map[crossline]
                    full_trace = f.trace.raw[trace_idx]
                    volume[i_idx, c_idx, :] = full_trace[sample_indices]
        print("Volume loaded successfully.")
        print(f"Volume shape: (Inlines, Crosslines, Samples) = {volume.shape}")
        return volume, np.array(inlines_to_load), np.array(crosslines_to_load), z_coords
    except Exception as e:
        print(f"An error occurred while loading the SEG-Y file: {e}")
        return None, None, None, None

# Step 4: Fungsi Pemrosesan Data (Versi CPU - NumPy)
def process_volume_cpu(volume, agc_window=200, clip_percent=0.98):
    """Menerapkan AGC dan clipping pada seluruh volume 3D menggunakan NumPy (CPU)."""
    if volume is None: return None
    print("Processing volume on CPU with NumPy...")
    processed_volume = np.zeros_like(volume)
    n_inlines = volume.shape[0]
    for i in range(n_inlines):
        inline_slice = volume[i, :, :].T
        agc_slice = apply_agc_cpu(inline_slice, window_length=agc_window)
        processed_volume[i, :, :] = agc_slice.T
    clipped_volume = apply_clipping_cpu(processed_volume, clip_percent)
    print("CPU processing complete.")
    return clipped_volume

def apply_agc_cpu(data: np.ndarray, window_length: int) -> np.ndarray:
    """Menerapkan Automatic Gain Control (AGC) pada data 2D menggunakan NumPy."""
    result = data.copy()
    n_samples, n_traces = data.shape
    half_window = window_length // 2
    for j in range(n_traces):
        for i in range(n_samples):
            start_idx = max(0, i - half_window)
            end_idx = min(n_samples, i + half_window + 1)
            window_data = data[start_idx:end_idx, j]
            rms = np.sqrt(np.mean(window_data**2))
            if rms > 1e-9:
                result[i, j] = data[i, j] / rms
    return result

def apply_clipping_cpu(data: np.ndarray, clip_percent: float) -> np.ndarray:
    """Menerapkan quantile clipping menggunakan NumPy."""
    lower_bound = np.percentile(data, (1 - clip_percent) / 2 * 100)
    upper_bound = np.percentile(data, (1 + clip_percent) / 2 * 100)
    return np.clip(data, lower_bound, upper_bound)


# Step 4.5: Fungsi Pemrosesan Data (Versi GPU - PyTorch)
def process_volume_gpu(volume, agc_window=200, clip_percent=0.98):
    """Menerapkan AGC dan clipping pada seluruh volume 3D menggunakan PyTorch (GPU)."""
    if volume is None: return None
    print("Processing volume on GPU with PyTorch...")
    volume_tensor = torch.from_numpy(volume).to(DEVICE)
    processed_tensor = torch.zeros_like(volume_tensor)
    n_inlines = volume_tensor.shape[0]
    for i in range(n_inlines):
        inline_slice = volume_tensor[i, :, :].T
        agc_slice = apply_agc_gpu(inline_slice, window_length=agc_window)
        processed_tensor[i, :, :] = agc_slice.T
    clipped_tensor = apply_clipping_gpu(processed_tensor, clip_percent)
    result_volume = clipped_tensor.cpu().numpy()
    print("GPU processing complete.")
    return result_volume

def apply_agc_gpu(tensor_slice, window_length):
    """Menerapkan AGC secara efisien pada slice 2D di GPU menggunakan PyTorch."""
    inp = tensor_slice.T.unsqueeze(1)
    avg_kernel = torch.ones(1, 1, window_length, device=DEVICE) / window_length
    squared_inp = inp**2
    mean_of_squares = F.conv1d(squared_inp, avg_kernel, padding='same')
    rms = torch.sqrt(mean_of_squares)
    rms[rms < 1e-9] = 1e-9
    agc_slice = inp / rms
    return agc_slice.squeeze(1).T

def apply_clipping_gpu(tensor, clip_percent):
    """Menerapkan quantile clipping di GPU menggunakan PyTorch."""
    lower_q = (1 - clip_percent) / 2.0
    upper_q = (1 + clip_percent) / 2.0
    lower_bound = torch.quantile(tensor, lower_q)
    upper_bound = torch.quantile(tensor, upper_q)
    return torch.clamp(tensor, lower_bound, upper_bound)


# Step 5: Fungsi untuk Menjalankan Aplikasi Web Dash
def run_dash_server(volume, inlines, crosslines, z_coords):
    """Membangun dan menjalankan server web Dash untuk visualisasi."""
    if not DASH_AVAILABLE:
        print("\n" + "="*50)
        print("❌ Error: Pustaka Dash tidak ditemukan.")
        print("Silakan instal dengan menjalankan: pip install dash")
        print("="*50 + "\n")
        return
        
    if volume is None:
        print("Cannot start server, volume data is missing.")
        return

    print("Generating 3D visualization figure...")
    # Gunakan koordinat sebenarnya yang dilewatkan sebagai argumen
    X, Y, Z = np.meshgrid(inlines, crosslines, z_coords, indexing='ij')
    
    fig = go.Figure(data=go.Volume(
        x=Y.flatten(), y=X.flatten(), z=Z.flatten(), # Tukar X dan Y jika perlu
        value=volume.flatten(),
        isomin=np.min(volume), isomax=np.max(volume),
        opacity=0.1, surface_count=15,
        colorscale='RdBu', reversescale=True,
        cmin=np.percentile(volume, 2), # Tingkatkan kontras
        cmax=np.percentile(volume, 98)
    ))
    
    fig.update_layout(
        title='Interactive 3D Seismic Cube',
        scene=dict(
            xaxis_title='Crossline', yaxis_title='Inline', zaxis_title='Time (ms)',
            aspectratio=dict(x=1, y=1, z=0.75),
            zaxis=dict(autorange='reversed') # Membalik sumbu Z untuk kedalaman/waktu
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # --- PERUBAHAN UTAMA: Membuat dan Menjalankan Aplikasi Dash ---
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("3D Seismic Volume Visualizer"),
        dcc.Graph(
            id='seismic-cube',
            figure=fig,
            style={'height': '80vh'} # Membuat grafik mengisi sebagian besar tinggi layar
        )
    ])
    
    # --- Dapatkan IP lokal untuk ditampilkan ke pengguna ---
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
    except:
        ip_address = "Tidak dapat mendeteksi IP"

    print("\n" + "="*50)
    print("🚀 Dash server is running and accessible on your network!")
    print(f"👉 Di komputer ini, buka: http://127.0.0.1:8050/")
    print(f"👉 Di perangkat lain (HP/laptop) di jaringan yang sama, buka: http://{ip_address}:8050/")
    print("   (Pastikan perangkat terhubung ke WiFi yang sama)")
    print("ℹ️  Untuk menghentikan server, tekan CTRL+C di terminal ini.")
    print("="*50 + "\n")
    
    # --- FIX: Menggunakan app.run() dan host='0.0.0.0' agar bisa diakses di jaringan ---
    app.run(debug=True, host='0.0.0.0')


# --- Blok Eksekusi Utama ---
def main():
    """Fungsi utama untuk menjalankan seluruh alur kerja."""
    start_total_time = time.time()
    
    seismic_volume, inlines, crosslines, z_coords = load_segy_volume(SEGY_PATH, MAX_INLINES, MAX_CROSSLINES, SUBSAMPLE)
    
    if seismic_volume is not None:
        start_process_time = time.time()
        
        if DEVICE.type == 'cuda':
            processed_seismic_volume = process_volume_gpu(seismic_volume)
        else:
            processed_seismic_volume = process_volume_cpu(seismic_volume)
            
        end_process_time = time.time()
        print(f"⏱️ Waktu pemrosesan data: {end_process_time - start_process_time:.2f} detik")
        
        # Panggil fungsi untuk menjalankan server Dash dengan data koordinat
        run_dash_server(processed_seismic_volume, inlines, crosslines, z_coords)
        
    end_total_time = time.time()
    print(f"⏱️ Total waktu eksekusi: {end_total_time - start_total_time:.2f} detik")


if __name__ == "__main__":
    main()

