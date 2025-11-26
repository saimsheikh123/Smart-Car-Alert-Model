import os
import random
import json
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
import torch

from ast_classifier import ASTClassifier

"""
Sample evaluation script:
- Loads the fine-tuned AST model
- For each test class folder, evaluates up to 10 audio files
- Prints per-file predictions and a summary (correct / incorrect / confusion targets)

Run:
  python evaluate_samples.py --model ../ast_6class_model --test-root ../train/dataset/test --samples 10
"""

import argparse

SEVERITY_CRITICAL = {"collision_sounds", "human_scream", "emergency_sirens"}
SEVERITY_IMPORTANT = {"alert_sounds"}
SEVERITY_LOW = {"environmental_sounds", "road_traffic"}

def compute_severity(pred_class: str, confidence: float):
    level = 1
    label = "ignore"
    justification = "Low confidence or low-importance class"
    if pred_class in SEVERITY_CRITICAL:
        if confidence >= 0.85:
            level, label = 5, "severe"
            justification = "High-confidence critical event"
        elif confidence >= 0.60:
            level, label = 4, "high"
            justification = "Likely critical event"
        elif confidence >= 0.40:
            level, label = 3, "warning"
            justification = "Possible critical event"
        else:
            level, label = 2, "info"
            justification = "Low-confidence critical class"
    elif pred_class in SEVERITY_IMPORTANT:
        if confidence >= 0.70:
            level, label = 3, "warning"
            justification = "Attention recommended"
        elif confidence >= 0.50:
            level, label = 2, "info"
            justification = "Moderate confidence alert sound"
    elif pred_class in SEVERITY_LOW:
        if confidence >= 0.70:
            level, label = 2, "info"
            justification = "Ambient/non-actionable"
    else:
        if confidence >= 0.60:
            level, label = 2, "info"
            justification = "Unmapped but confident"
    return {"level_num": level, "level": label, "justification": justification}

def load_audio(path: Path, target_sr=16000, max_len_s=5):
    data, sr = sf.read(str(path))
    if sr != target_sr:
        data = librosa.resample(data.astype("float32"), orig_sr=sr, target_sr=target_sr)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    max_samples = max_len_s * target_sr
    if len(data) > max_samples:
        data = data[:max_samples]
    else:
        data = np.pad(data, (0, max_samples - len(data)))
    import io
    import soundfile as sf2
    buf = io.BytesIO()
    sf2.write(buf, data, target_sr, format='WAV')
    return buf.getvalue()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../ast_6class_model', help='Path to fine-tuned AST model directory')
    parser.add_argument('--test-root', type=str, default='../train/dataset/test', help='Root of test dataset class folders')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples per class (max)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    test_root = Path(args.test_root).resolve()
    if not test_root.exists():
        print(f"[ERROR] Test root not found: {test_root}")
        return

    device = 'cpu'
    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        print(f"[WARN] Fine-tuned model dir not found: {model_dir}; falling back to pretrained aggregation")
        model_source = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    else:
        model_source = str(model_dir)

    classifier = ASTClassifier(model_source, device=device)

    class_dirs = [d for d in test_root.iterdir() if d.is_dir()]
    class_dirs.sort()
    results = {}

    print("\nPer-class sampled evaluation (up to", args.samples, "files each)\n")
    for d in class_dirs:
        cls_name = d.name
        wavs = list(d.rglob('*.wav'))
        if not wavs:
            print(f"[SKIP] {cls_name}: no .wav files")
            continue
        random.shuffle(wavs)
        subset = wavs[:args.samples]
        per_files = []
        correct = 0
        collision_mis_as_road = 0
        for p in subset:
            try:
                audio_bytes = load_audio(p)
                pred = classifier.predict(audio_bytes)
                pred_class = pred.get('class', 'unknown')
                conf = float(pred.get('confidence', 0.0))
                severity = compute_severity(pred_class, conf)
                # Map any glass breaking style folder to collision_sounds taxonomy
                mapped_true = cls_name
                if 'glass' in cls_name.lower():
                    mapped_true = 'collision_sounds'
                is_correct = (pred_class == mapped_true)
                if mapped_true == 'collision_sounds' and pred_class == 'road_traffic':
                    collision_mis_as_road += 1
                if is_correct:
                    correct += 1
                per_files.append({
                    'file': p.name,
                    'true': cls_name,
                    'mapped_true': mapped_true,
                    'pred': pred_class,
                    'confidence': round(conf, 4),
                    'severity': severity,
                    'correct': is_correct
                })
            except Exception as e:
                per_files.append({'file': p.name, 'error': str(e)})

        acc = correct / len(per_files) if per_files else 0.0
        results[cls_name] = {
            'total_sampled': len(per_files),
            'correct': correct,
            'accuracy': round(acc, 3),
            'collision_misclassified_as_road_traffic': collision_mis_as_road if ('glass' in cls_name.lower() or cls_name == 'collision_sounds') else None,
            'samples': per_files
        }
        print(f"Class {cls_name:20} | Acc {acc*100:5.1f}% | n={len(per_files)}")
        if ('glass' in cls_name.lower() or cls_name == 'collision_sounds') and collision_mis_as_road > 0:
            print(f"  -> {collision_mis_as_road} collision sample(s) predicted as road_traffic")

    # Summary
    print("\nDetailed JSON summary (truncated to correctness counts):")
    summary = {c: {k: v for k, v in r.items() if k in ('total_sampled','correct','accuracy','collision_misclassified_as_road_traffic')} for c, r in results.items()}
    print(json.dumps(summary, indent=2))

    # Save full results
    out_path = Path('sample_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull per-file results saved to {out_path.resolve()}")

if __name__ == '__main__':
    main()
