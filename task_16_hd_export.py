import os
import pandas as pd
import numpy as np
import logging
import openslide
import scipy.ndimage as ndimage
import tifffile
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
INVENTORY_CSV = os.path.join(BASE_DIR, "output", "task_16_pathology_inventory.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "pathology_raw_hd")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_pathology_path(patient_id):
    if not os.path.exists(INVENTORY_CSV): return None
    inv = pd.read_csv(INVENTORY_CSV, dtype={'Patient_ID': str})
    row = inv[inv['Patient_ID'] == str(patient_id)]
    if not row.empty: return row.iloc[0]['Path']
    return None

def export_full_res_chunked(path_file, patient_id):
    try:
        logging.info(f"Opening slide: {os.path.basename(path_file)}")
        slide = openslide.OpenSlide(path_file)
        
        # 1. Detect ROI (Same logic as before)
        level_idx = min(3, slide.level_count - 1)
        w_low, h_low = slide.level_dimensions[level_idx]
        low_res = np.array(slide.read_region((0, 0), level_idx, (w_low, h_low)).convert("L"))
        mask = (low_res < 230) & (low_res > 20)
        
        labeled, n = ndimage.label(mask)
        if n == 0:
            w0, h0 = slide.level_dimensions[0]
            roi_0 = (int(w0*0.2), int(h0*0.2), int(w0*0.6), int(h0*0.6))
        else:
            coords = np.argwhere(mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            ds = slide.level_downsamples[level_idx]
            roi_0 = (int(x_min * ds), int(y_min * ds), int((x_max - x_min) * ds), int((y_max - y_min) * ds))

        x_start, y_start, w_target, h_target = roi_0
        logging.info(f"Exporting ROI: {w_target}x{h_target} pixels.")
        
        # 2. Setup Output File
        out_path = os.path.join(OUTPUT_DIR, f"Patho_{patient_id}_FullRes.tif")
        
        # Define Tiff Parameters
        TILE_SIZE = 4096
        
        # Use tifffile to write tiles. 
        # Writing a huge contiguous array via memmap is tricky with BigTIFF compression.
        # Instead, we use tifffile.TiffWriter which supports writing tiles directly.
        
        with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
            
            # We can't easily write a single huge array if we don't have it in memory.
            # But TiffWriter allows writing tiles.
            # However, standard image viewers expect a single image.
            # The most robust way for HUGE images with low RAM is to write as TILES.
            
            # Calculate grid
            nx = int(np.ceil(w_target / TILE_SIZE))
            ny = int(np.ceil(h_target / TILE_SIZE))
            
            logging.info(f"Grid: {nx} x {ny} tiles ({nx*ny} total)")
            
            # Generator to yield tiles
            def tile_generator():
                for y in range(ny):
                    for x in range(nx):
                        # Tile Coordinates in Level 0
                        tx = x_start + x * TILE_SIZE
                        ty = y_start + y * TILE_SIZE
                        
                        # Actual dimensions of this tile (handle edges)
                        tw = min(TILE_SIZE, x_start + w_target - tx)
                        th = min(TILE_SIZE, y_start + h_target - ty)
                        
                        # Read
                        # Note: openslide read_region might return RGBA, convert to RGB
                        tile = slide.read_region((tx, ty), 0, (tw, th)).convert("RGB")
                        tile_arr = np.array(tile)
                        
                        # Pad if edge tile is smaller than TILE_SIZE (required for consistent tile size in some Tiff modes)
                        # But TiffWriter with 'tile' option usually handles fixed grid.
                        # Let's check tifffile docs: It expects tiles to be passed sequentially.
                        # We must pad to TILE_SIZE if we declare tile size.
                        if tw < TILE_SIZE or th < TILE_SIZE:
                            pad_arr = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) + 255 # White padding
                            pad_arr[:th, :tw, :] = tile_arr
                            yield pad_arr
                        else:
                            yield tile_arr
                            
            # Write using tiles
            # This creates a tiled TIFF which is standard for WSI
            tif.write(
                tile_generator(),
                shape=(h_target, w_target, 3),
                dtype=np.uint8,
                tile=(TILE_SIZE, TILE_SIZE),
                compression=None, # No compression for speed
                photometric='rgb'
            )
            
        logging.info(f"Success: {out_path}")
        return True

    except Exception as e:
        logging.error(f"Failed {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

import multiprocessing

def process_wrapper(pid):
    path = get_pathology_path(pid)
    if path: 
        export_full_res_chunked(path, pid)

def main():
    target_ids = ['2020027543', '2120033358', '2120036947']
    
    # Use 3 processes for 3 patients
    with multiprocessing.Pool(processes=3) as pool:
        pool.map(process_wrapper, target_ids)

if __name__ == "__main__":
    main()