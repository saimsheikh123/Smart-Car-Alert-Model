import os
import json
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa

from ast_classifier import ASTClassifier

"""Quick audit script for collision_sounds test set.
Usage:
  python verify_collision_samples.py --model ../ast_6class_model --test-root ../train/dataset/test --limit 200

Sets COLLISION_HEURISTIC=1 automatically to apply override heuristic.
Outputs JSON summary and prints any non-collision predictions.
"""

import argparse

def load_audio(path, target_sr=16000, max_len_s=10):
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
    import io, soundfile as sf2
    buf = io.BytesIO()
    sf2.write(buf, audio, sr, format='WAV')
    return buf.getvalue()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='../ast_6class_model')
    ap.add_argument('--test-root', default='../train/dataset/test')
    ap.add_argument('--limit', type=int, default=200, help='Max files to scan')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.environ['COLLISION_HEURISTIC'] = '1'

    test_root = Path(args.test_root).resolve()
    coll_dir = test_root / 'collision_sounds'
    if not coll_dir.exists():
        print(f"[ERROR] collision_sounds dir missing: {coll_dir}")
        return

    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        print(f"[WARN] Fine-tuned model not found; will use fallback mapping")
    clf = ASTClassifier(str(model_dir), device='cpu')

    wavs = list(coll_dir.rglob('*.wav'))
    wavs.sort()
    if args.limit:
        wavs = wavs[:args.limit]

    summary = {
        'total': 0,
        'collision_as_collision': 0,
        'collision_as_other': 0,
        'override_used': 0,
        'details': []
    }

    for w in wavs:
        try:
            audio_bytes = load_audio(w)
            pred = clf.predict(audio_bytes)
            pred_class = pred['class']
            override = bool(pred['all_probs'].get('_collision_override'))
            summary['total'] += 1
            if pred_class == 'collision_sounds':
                summary['collision_as_collision'] += 1
            else:
                summary['collision_as_other'] += 1
            if override:
                summary['override_used'] += 1
            summary['details'].append({
                'file': w.name,
                'pred': pred_class,
                'confidence': round(pred['confidence'], 4),
                'top2': pred['top2'],
                'override': override
            })
        except Exception as e:
            summary['details'].append({'file': w.name, 'error': str(e)})

    # Print non-collision predictions
    bad = [d for d in summary['details'] if d.get('pred') != 'collision_sounds' and 'error' not in d]
    if bad:
        print("\nNon-collision predictions (showing up to 30):")
        for b in bad[:30]:
            print(f"  {b['file']:<25} -> {b['pred']:<18} conf={b['confidence']:.3f} override={b['override']}")
    else:
        print("\nAll scanned collision samples predicted correctly.")

    acc = (summary['collision_as_collision'] / summary['total']) if summary['total'] else 0.0
    print(f"\nSummary: {summary['collision_as_collision']}/{summary['total']} correct ({acc*100:.1f}%), overrides used: {summary['override_used']}")

    out_path = Path('collision_audit.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved detailed audit to {out_path.resolve()}")

if __name__ == '__main__':
    main()
