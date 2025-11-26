"""Test script for 6-class fine-tuned AST model.

Ensures we are using the local fine-tuned directory `../ast_6class_model`.
Reports per-class accuracy (up to a configurable sample limit) and flags
any collision_sounds samples misclassified as road_traffic.

Usage (PowerShell):
  cd Audio_Models
  set FORCE_CPU=1
  python test_6class.py --model ../ast_6class_model --data-root ../train/dataset/test --limit 50

Options:
  --limit N   Max files per class (default 30)
  --seed S    Reproducible shuffle seed
  --show-all  Print every file result (else only mistakes)

If the model directory is missing, the script will EXIT instead of
falling back to the pretrained model (to avoid confusing results).
"""

from pathlib import Path
import argparse
import random
import numpy as np
import soundfile as sf
import io
import json
import librosa
import os

from ast_classifier import ASTClassifier

CLASSES = [
    "alert_sounds",
    "collision_sounds",
    "emergency_sirens",
    "environmental_sounds",
    "human_scream",
    "road_traffic",
]

def load_audio_bytes(path: Path, target_sr=16000, max_len_s=10):
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr
    max_samples = max_len_s * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='../ast_6class_model')
    ap.add_argument('--data-root', default='../train/dataset/test')
    ap.add_argument('--limit', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--show-all', action='store_true', help='Print every file instead of only mistakes')
    ap.add_argument('--enable-collision-heuristic', action='store_true', help='Turn on COLLISION_HEURISTIC override')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        print(f"[FATAL] Fine-tuned model directory not found: {model_dir}")
        print("Ensure you have trained and saved the AST model locally before running this test.")
        raise SystemExit(2)

    if args.enable_collision_heuristic:
        os.environ['COLLISION_HEURISTIC'] = '1'
        print('[INFO] COLLISION_HEURISTIC enabled.')

    clf = ASTClassifier(str(model_dir), device='cpu')

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"[FATAL] Data root does not exist: {data_root}")
        raise SystemExit(3)

    print("="*80)
    print("6-CLASS MODEL TEST (Fine-Tuned AST)")
    print("Model directory:", model_dir)
    print("Data root:", data_root)
    print("Limit per class:", args.limit)
    print("Collision heuristic:", 'ON' if args.enable_collision_heuristic else 'OFF')
    print("="*80)

    report = {}
    collision_mis_as_traffic = 0

    for cls in CLASSES:
        cls_dir = data_root / cls
        if not cls_dir.exists():
            print(f"[WARN] Missing class directory: {cls_dir}")
            continue
        wavs = list(cls_dir.rglob('*.wav'))
        if not wavs:
            print(f"[WARN] No wav files for class {cls}")
            continue
        random.shuffle(wavs)
        subset = wavs[:args.limit]
        correct = 0
        details = []
        for w in subset:
            try:
                audio_bytes = load_audio_bytes(w)
                pred = clf.predict(audio_bytes)
                pred_class = pred['class']
                conf = pred['confidence']
                is_correct = (pred_class == cls)
                if cls == 'collision_sounds' and pred_class == 'road_traffic':
                    collision_mis_as_traffic += 1
                if is_correct:
                    correct += 1
                entry = {
                    'file': w.name,
                    'true': cls,
                    'pred': pred_class,
                    'confidence': round(conf, 4),
                    'top2': pred['top2'],
                    'override': bool(pred['all_probs'].get('_collision_override')),
                    'correct': is_correct
                }
                details.append(entry)
            except Exception as e:
                details.append({'file': w.name, 'error': str(e), 'correct': False})

        acc = correct / len(details) if details else 0.0
        report[cls] = {
            'sampled': len(details),
            'correct': correct,
            'accuracy': round(acc, 3),
            'samples': details
        }
        print(f"Class {cls:18} Acc {acc*100:5.1f}% ({correct}/{len(details)})")
        if cls == 'collision_sounds' and collision_mis_as_traffic > 0:
            print(f"  -> {collision_mis_as_traffic} collision sample(s) predicted as road_traffic")

        if not args.show_all:
            mistakes = [d for d in details if not d.get('correct') and 'error' not in d]
            for m in mistakes[:10]:
                print(f"    MIS: {m['file']:<22} -> {m['pred']:<16} conf={m['confidence']:.3f}")

    print("="*80)
    print("RESULT SUMMARY")
    for cls, stats in report.items():
        print(f"  {cls:18} {stats['correct']:>3}/{stats['sampled']:<3} acc={stats['accuracy']:.3f}")
    print(f"Collision misclassified as road_traffic: {collision_mis_as_traffic}")
    print("="*80)

    out_path = Path('test_6class_results.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Detailed JSON saved to {out_path.resolve()}")

if __name__ == '__main__':
    main()
