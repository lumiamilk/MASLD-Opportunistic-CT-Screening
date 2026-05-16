import os
import csv
import sys
from collections import defaultdict

def get_file_info(path):
    try:
        stat = os.stat(path)
        return stat.st_ino, stat.st_dev
    except OSError:
        return None, None

def find_hardlink_sources(cleaned_dir, source_dir, output_file):
    print(f"Scanning {cleaned_dir}...")
    
    # Map: (device, inode) -> {'cleaned': [], 'original': []}
    inode_map = defaultdict(lambda: {'cleaned': [], 'original': []})
    
    # Step 1: Scan CT_Cleaned
    cleaned_count = 0
    for root, dirs, files in os.walk(cleaned_dir):
        for name in files:
            path = os.path.join(root, name)
            inode, dev = get_file_info(path)
            if inode is not None:
                inode_map[(dev, inode)]['cleaned'].append(path)
                cleaned_count += 1
                if cleaned_count % 1000 == 0:
                     print(f"Scanned {cleaned_count} files in Cleaned...", end='\r')
    print(f"Scanned {cleaned_count} files in Cleaned. Found {len(inode_map)} unique inodes.")

    # Step 2: Scan CT_origianl_data_2018_2025
    print(f"Scanning {source_dir}...")
    source_count = 0
    matched_inodes = 0
    
    # Optimization: We only care about inodes that exist in Cleaned
    target_inodes = set(inode_map.keys())
    
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            path = os.path.join(root, name)
            inode, dev = get_file_info(path)
            
            if (dev, inode) in target_inodes:
                inode_map[(dev, inode)]['original'].append(path)
                matched_inodes += 1
            
            source_count += 1
            if source_count % 1000 == 0:
                print(f"Scanned {source_count} files in Source. Matched {matched_inodes}...", end='\r')

    print(f"Scanned {source_count} files in Source. Total matches found: {matched_inodes}")

    # Step 3: Output results
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['MFT_Device', 'MFT_Inode', 'Cleaned_Paths', 'Original_Paths'])
        
        for (dev, inode), data in inode_map.items():
            cleaned_paths = ';'.join(data['cleaned'])
            original_paths = ';'.join(data['original'])
            writer.writerow([dev, inode, cleaned_paths, original_paths])

    print("Done.")

if __name__ == "__main__":
    cleaned_dir = r"D:\mWork\paper0\data\CT_Cleaned"
    source_dir = r"D:\mWork\paper0\data\CT_origianl_data_2018_2025"
    output_file = r"D:\mWork\paper0\output\task_5.6_hardlink_source_map.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    find_hardlink_sources(cleaned_dir, source_dir, output_file)
