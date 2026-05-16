import os
from PIL import Image, ImageDraw, ImageFont
import logging

# --- Configuration ---
BASE_DIR = os.getcwd()
if "/mnt/d" not in BASE_DIR and "paper0" not in BASE_DIR:
     BASE_DIR = "/mnt/d/mWork/paper0"

INPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "thesis_assets", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input Files
IMG_A = os.path.join(INPUT_DIR, "task17.5.1_CT_Original.png")
IMG_B = os.path.join(INPUT_DIR, "task17.5.1_HE_ROI_2.jpg")
IMG_C = os.path.join(INPUT_DIR, "task17.5.1_HE_Mask_2.jpg")
IMG_D = os.path.join(INPUT_DIR, "task17.5.1_Ret_ROI_2.jpg")

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Fig_6_1_Multimodal_Panel_Final.png")

# Settings
TARGET_SIZE = (1000, 1000)
DPI = 300
GAP = 40  # Gap between images
MARGIN = 40 # Margin around the whole panel
LABEL_BOX_SIZE = 80
CAPTION_HEIGHT = 80 # Extra space below image for caption

# Fonts
# Try to find a nice font
FONT_PATH = None
POSSIBLE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:/Windows/Fonts/Arialbd.ttf", # Windows
    "Arial Bold.ttf"
]

for fp in POSSIBLE_FONTS:
    if os.path.exists(fp):
        FONT_PATH = fp
        break

def create_labeled_tile(img_path, label_char, caption_text, font_label, font_caption):
    # 1. Load and Resize
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        # Create placeholder
        img = Image.new('RGB', TARGET_SIZE, color=(200, 200, 200))

    # 2. Create Canvas with space for caption
    # Canvas size: Width = 1000, Height = 1000 + CAPTION_HEIGHT
    tile_w, tile_h = TARGET_SIZE
    canvas_h = tile_h + CAPTION_HEIGHT
    canvas = Image.new("RGB", (tile_w, canvas_h), "white")
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # 3. Draw Label Box (A, B...) Top-Left
    # Box style: White background, slight transparency? Or solid white to be clear.
    # User requested: "black/white background small box". Let's do White Box with Black Text for high contrast.
    
    box_margin = 0
    box_w, box_h = LABEL_BOX_SIZE, LABEL_BOX_SIZE
    # Draw solid white rectangle
    draw.rectangle([box_margin, box_margin, box_margin+box_w, box_margin+box_h], fill="white", outline="black", width=3)
    
    # Draw Text centered in box
    text_bbox = draw.textbbox((0, 0), label_char, font=font_label)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    x_pos = box_margin + (box_w - text_w) / 2
    # Adjust y_pos slightly for visual centering
    y_pos = box_margin + (box_h - text_h) / 2 - 5 
    
    draw.text((x_pos, y_pos), label_char, fill="black", font=font_label)

    # 4. Draw Caption at Bottom Center (in the white space)
    cap_bbox = draw.textbbox((0, 0), caption_text, font=font_caption)
    cap_w = cap_bbox[2] - cap_bbox[0]
    cap_h = cap_bbox[3] - cap_bbox[1]
    
    cap_x = (tile_w - cap_w) / 2
    cap_y = tile_h + (CAPTION_HEIGHT - cap_h) / 2 - 5
    
    draw.text((cap_x, cap_y), caption_text, fill="black", font=font_caption)

    return canvas

def main():
    print("Starting Panel Generation...")
    
    # Load Fonts
    try:
        if FONT_PATH:
            font_label = ImageFont.truetype(FONT_PATH, 60)
            font_caption = ImageFont.truetype(FONT_PATH, 40)
            print(f"Using font: {FONT_PATH}")
        else:
            font_label = ImageFont.load_default()
            font_caption = ImageFont.load_default()
            print("Using default font (system font not found).")
    except Exception as e:
        print(f"Font loading error: {e}")
        font_label = ImageFont.load_default()
        font_caption = ImageFont.load_default()

    # Generate Tiles
    tile_a = create_labeled_tile(IMG_A, "A", "CT Scan (Liver Window)", font_label, font_caption)
    tile_b = create_labeled_tile(IMG_B, "B", "Pathology (H&E)", font_label, font_caption)
    tile_c = create_labeled_tile(IMG_C, "C", "AI Quantification (Fat Mask)", font_label, font_caption)
    tile_d = create_labeled_tile(IMG_D, "D", "Reticulin Stain (Fibrosis)", font_label, font_caption)

    # Calculate Total Size
    # 2 cols, 2 rows
    # Width = MARGIN + TileW + GAP + TileW + MARGIN
    # Height = MARGIN + TileH + GAP + TileH + MARGIN
    
    tile_w, tile_h = tile_a.size # Note tile_h includes caption space
    
    total_w = MARGIN * 2 + tile_w * 2 + GAP
    total_h = MARGIN * 2 + tile_h * 2 + GAP
    
    final_img = Image.new("RGB", (total_w, total_h), "white")
    
    # Paste Tiles
    # (0, 0) -> A
    pos_a = (MARGIN, MARGIN)
    # (0, 1) -> B
    pos_b = (MARGIN + tile_w + GAP, MARGIN)
    # (1, 0) -> C
    pos_c = (MARGIN, MARGIN + tile_h + GAP)
    # (1, 1) -> D
    pos_d = (MARGIN + tile_w + GAP, MARGIN + tile_h + GAP)
    
    final_img.paste(tile_a, pos_a)
    final_img.paste(tile_b, pos_b)
    final_img.paste(tile_c, pos_c)
    final_img.paste(tile_d, pos_d)
    
    # Save
    final_img.save(OUTPUT_FILE, dpi=(DPI, DPI))
    print(f"Panel saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
