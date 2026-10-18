import os
import json
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Paths
GENERATED_AUDIO_DIR = "data/generated/audiogen/generated_audio"
GROUND_TRUTH_MANIFEST = "data/processed/gold_standard_50_sed_timestamps_populated.json"
OUTPUT_PREDICTED_MANIFEST = "data/processed/audiogen_predicted_sed_timestamps.json"

CONFIDENCE_THRESHOLD = 0.20

def load_yamnet_model():
    print("[*] Loading pre-trained YAMNet model from TFHub...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Load class map names
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csv_file:
        import csv
        reader = csv.DictReader(csv_file)
        for row in reader:
            class_names.append(row['display_name'])
            
    return model, np.array(class_names)

def process_wav(model, class_names, wav_path):
    # Load and resample to 16kHz
    audio_data, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    # Run YAMNet frame-level inference
    scores, embeddings, spectrogram = model(audio_data)
    scores = scores.numpy()  # (N_frames, 521)
    
    detected_events = []
    num_frames = scores.shape[0]

    for frame_idx in range(num_frames):
        t_start = round(frame_idx * 0.48, 2)
        t_end = round(t_start + 0.96, 2)

        top_class_idx = np.argmax(scores[frame_idx])
        confidence = float(scores[frame_idx][top_class_idx])

        if confidence >= CONFIDENCE_THRESHOLD:
            label = class_names[top_class_idx]
            
            if detected_events and detected_events[-1]["label"] == label:
                detected_events[-1]["t_end"] = t_end
            else:
                detected_events.append({
                    "label": label,
                    "confidence": round(confidence, 3),
                    "t_start": t_start,
                    "t_end": t_end
                })

    return detected_events

def run_generated_sed_extraction():
    if not os.path.exists(GROUND_TRUTH_MANIFEST):
        print(f"[!] Ground truth manifest missing: {GROUND_TRUTH_MANIFEST}")
        return

    with open(GROUND_TRUTH_MANIFEST, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    model, class_names = load_yamnet_model()
    predicted_manifest = []

    print(f"[*] Starting YAMNet SED timestamp extraction on generated audio clips...")

    for entry in ground_truth:
        sample_id = entry["sample_id"]
        caption = entry["caption"]
        
        # Audio file naming convention matching our Colab export
        wav_filename = f"sample_{sample_id:02d}_audiogen.wav"
        wav_path = os.path.join(GENERATED_AUDIO_DIR, wav_filename)

        if not os.path.exists(wav_path):
            print(f" -> [Skipped] Missing audio file: {wav_filename}")
            continue

        events = process_wav(model, class_names, wav_path)
        
        predicted_manifest.append({
            "sample_id": sample_id,
            "caption": caption,
            "ground_truth_events": entry.get("detected_events", []),
            "generated_predicted_events": events
        })
        print(f" -> Sample {sample_id:02d}: Extracted {len(events)} predicted event intervals.")

    with open(OUTPUT_PREDICTED_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(predicted_manifest, f, indent=4)

    print(f"\n[✓] Extracted predicted SED timestamps saved to: {OUTPUT_PREDICTED_MANIFEST}")

if __name__ == "__main__":
    run_generated_sed_extraction()