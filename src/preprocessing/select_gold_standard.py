import csv
import json
import random

INPUT_FILE = "data/processed/audiocaps_temporal_subset.csv"  # Test subset output
OUTPUT_CSV = "data/processed/gold_standard_50.csv"
OUTPUT_JSON = "data/processed/gold_standard_50.json"

# Set seed for exact academic reproducibility
SEED = 42
NUM_SAMPLES = 50

def curate_gold_standard():
    random.seed(SEED)
    
    with open(INPUT_FILE, mode='r', encoding='utf-8') as infile:
        reader = list(csv.DictReader(infile))
        
    print(f"[*] Total available candidate prompts: {len(reader)}")
    
    # Stratify/sample to ensure diversity across conjunctions
    categories = {'then': [], 'after': [], 'before': [], 'while': [], 'followed_by': []}
    
    for row in reader:
        caption = row.get('caption', '').lower()
        if 'followed by' in caption:
            categories['followed_by'].append(row)
        elif 'then' in caption:
            categories['then'].append(row)
        elif 'after' in caption:
            categories['after'].append(row)
        elif 'while' in caption:
            categories['while'].append(row)
        elif 'before' in caption:
            categories['before'].append(row)
            
    selected_samples = []
    
    # Draw balanced samples across categories
    samples_per_cat = NUM_SAMPLES // len(categories)
    for cat, items in categories.items():
        count = min(len(items), samples_per_cat)
        selected_samples.extend(random.sample(items, count))
        
    # Fill remaining quota if needed
    if len(selected_samples) < NUM_SAMPLES:
        remaining_pool = [r for r in reader if r not in selected_samples]
        needed = NUM_SAMPLES - len(selected_samples)
        selected_samples.extend(random.sample(remaining_pool, needed))
        
    # Write to CSV
    fieldnames = list(reader[0].keys())
    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_samples)
        
    # Write to structured JSON for downstream SED timestamp mapping
    with open(OUTPUT_JSON, mode='w', encoding='utf-8') as json_out:
        json.dump(selected_samples, json_out, indent=4)
        
    print(f"[✓] Successfully exported {len(selected_samples)} Gold Standard prompts!")
    print(f"[✓] Saved to: {OUTPUT_CSV} and {OUTPUT_JSON}")

if __name__ == "__main__":
    curate_gold_standard()