import sqlite3
import pandas as pd
import os
import sys
import logging

# Configuration
BASE_DIR = r"D:\\mWork\\paper0"
DB_PATH = os.path.join(BASE_DIR, "output", "task_1_ct_metadata.db")
REPORT_PATH = os.path.join(BASE_DIR, "output", "task_1_analysis_report_v2.md")
LOG_PATH = os.path.join(BASE_DIR, "output", "task_1_v2_log.txt")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def generate_correlation_report():
    if not os.path.exists(DB_PATH):
        logging.error(f"Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 1. Master Grouping Query
        # We group by the physical/protocol attributes to see unique "Scan Protocols"
        query = """
        SELECT 
            series_description, 
            kernel, 
            slice_thickness, 
            kvp,
            COUNT(*) as series_count,
            MIN(image_count) as min_imgs,
            MAX(image_count) as max_imgs,
            ROUND(AVG(image_count), 1) as avg_imgs
        FROM ct_series 
        GROUP BY series_description, kernel, slice_thickness, kvp
        ORDER BY series_count DESC
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Generate Markdown Report
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("# Task 1 Analysis V2: Protocol Correlation Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            f.write("This report correlates `SeriesDescription`, `Kernel`, `SliceThickness`, and `KVP` to identify distinct scanning protocols and their frequency.\n\n")
            
            # --- Section 1: Top 20 Protocols ---
            f.write("## 1. Top 20 Most Common Protocols (Combinations)\n")
            f.write("A 'Protocol' here is defined as a unique combination of Description, Kernel, Thickness, and KVP.\n\n")
            
            # Format columns for display (handle missing values)
            display_df = df.copy()
            display_df['kernel'] = display_df['kernel'].fillna('(Empty)')
            display_df['series_description'] = display_df['series_description'].fillna('(Empty)')
            display_df['slice_thickness'] = display_df['slice_thickness'].fillna('N/A')
            display_df['kvp'] = display_df['kvp'].fillna('N/A')
            
            top_20 = display_df.head(20)
            f.write(top_20.to_markdown(index=False))
            f.write("\n\n")
            
            # --- Section 2: Kernel Consistency by Description ---
            f.write("## 2. Kernel Consistency Analysis\n")
            f.write("For specific Series Descriptions, what Kernels are used?\n\n")
            
            # Get top 10 descriptions
            top_desc = df.groupby('series_description')['series_count'].sum().nlargest(10).index.tolist()
            
            for desc in top_desc:
                if not desc: continue
                subset = display_df[display_df['series_description'] == desc]
                # Aggregate to show kernels for this description
                kernel_counts = subset.groupby(['kernel', 'slice_thickness'])['series_count'].sum().reset_index().sort_values('series_count', ascending=False)
                
                f.write(f"### Description: `{desc}`\n")
                f.write(kernel_counts.to_markdown(index=False))
                f.write("\n\n")

            # --- Section 3: Thickness Consistency ---
            f.write("## 3. Slice Thickness Analysis\n")
            f.write("Grouping by Thickness to see associated Kernels and Descriptions (Top 5 per thickness).\n\n")
            
            thicknesses = display_df['slice_thickness'].unique()
            thicknesses = sorted([x for x in thicknesses if x != 'N/A'])
            
            for th in thicknesses:
                subset = display_df[display_df['slice_thickness'] == th]
                total_for_th = subset['series_count'].sum()
                
                f.write(f"### Thickness: {th} mm (Total Series: {total_for_th})\n")
                top_configs = subset[['series_description', 'kernel', 'series_count']].head(5)
                f.write(top_configs.to_markdown(index=False))
                f.write("\n\n")

            # --- Section 4: Outlier Analysis (Image Counts) ---
            f.write("## 4. Image Count Variance in Protocols\n")
            f.write("Protocols with high variance in image count (Max - Min > 50). This might indicate variable scan lengths or mixed data types.\n\n")
            
            # Filter for variance
            high_var = display_df[display_df['max_imgs'] - display_df['min_imgs'] > 50].copy()
            high_var['variance'] = high_var['max_imgs'] - high_var['min_imgs']
            high_var = high_var.sort_values('series_count', ascending=False).head(20)
            
            if not high_var.empty:
                f.write(high_var[['series_description', 'kernel', 'slice_thickness', 'series_count', 'min_imgs', 'max_imgs']].to_markdown(index=False))
            else:
                f.write("No protocols found with significant image count variance (>50 images).")
            f.write("\n\n")

        logging.info(f"Report generated successfully at {REPORT_PATH}")
        conn.close()

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    generate_correlation_report()
