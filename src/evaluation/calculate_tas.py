import json
import numpy as np

MANIFEST_PATH = "data/processed/audiogen_predicted_sed_timestamps.json"

def calculate_iou(interval_a, interval_b):
    """Calculates Intersection over Union (IoU) between two temporal intervals."""
    start_a, end_a = interval_a
    start_b, end_b = interval_b
    
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    
    return intersection / union if union > 0 else 0.0

def evaluate_temporal_alignment():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    sample_scores = []
    
    print("=" * 65)
    print("      TEMPORAL ALIGNMENT SCORE (TAS) EVALUATION - AUDIOGEN      ")
    print("=" * 65)
    
    for item in data:
        sample_id = item["sample_id"]
        caption = item["caption"]
        gt_events = item.get("ground_truth_events", [])
        pred_events = item.get("generated_predicted_events", [])
        
        # Skip evaluating clips where ground truth SED had no events detected
        if not gt_events or not pred_events:
            continue
            
        # Match sound class events between Ground Truth and Generated predictions
        matches = 0
        total_gt = len(gt_events)
        iou_scores = []
        
        for gt in gt_events:
            gt_label = gt["label"].lower()
            gt_interval = (gt["t_start"], gt["t_end"])
            
            # Find matching class in prediction
            for pred in pred_events:
                pred_label = pred["label"].lower()
                if gt_label in pred_label or pred_label in gt_label:
                    matches += 1
                    pred_interval = (pred["t_start"], pred["t_end"])
                    iou = calculate_iou(gt_interval, pred_interval)
                    iou_scores.append(iou)
                    break
                    
        # Precision of class occurrence + Temporal IoU
        presence_score = matches / total_gt if total_gt > 0 else 0
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0
        
        # Combined TAS metric (0.0 to 1.0)
        tas_score = round(0.5 * presence_score + 0.5 * mean_iou, 4)
        sample_scores.append(tas_score)
        
        print(f"Sample {sample_id:02d} | TAS: {tas_score:.4f} | GT Events: {total_gt} | Pred Events: {len(pred_events)}")

    overall_tas = np.mean(sample_scores) if sample_scores else 0.0
    
    print("-" * 65)
    print(f"[✓] Overall AudioGen Baseline Temporal Alignment Score (TAS): {overall_tas:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_temporal_alignment()