import os
import json
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
import argparse
from ast_classifier import ASTClassifier

TARGET_CLASSES = ["collision_sounds", "road_traffic", "emergency_sirens"]

def load_audio(path: Path, target_sr=16000, max_len_s=10):
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
    max_samples = max_len_s * target_sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    import io, soundfile as sf2
    buf = io.BytesIO()
    sf2.write(buf, audio, target_sr, format='WAV')
    return buf.getvalue()

def is_extremely_off(pred_class: str, confidence: float, expected: str, all_probs: dict) -> bool:
    """Heuristic: consider sample extremely off if predicted != expected AND
    predicted confidence >= 0.60 AND expected class probability <= 0.05."""
    if pred_class == expected:
        return False
    exp_prob = all_probs.get(expected, 0.0)
    if confidence >= 0.60 and exp_prob <= 0.05:
        return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Audit key test classes and flag extremely off samples.")
    ap.add_argument('--model', default='../ast_6class_model')
    ap.add_argument('--test-root', default='../train/dataset/test')
    ap.add_argument('--output', default='key_class_audit.json')
    ap.add_argument('--enable-collision-heuristic', action='store_true')
    args = ap.parse_args()

    if args.enable_collision_heuristic:
        os.environ['COLLISION_HEURISTIC'] = '1'

    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        print(f"[FATAL] Fine-tuned model directory missing: {model_dir}")
        return
    clf = ASTClassifier(str(model_dir), device='cpu')

    test_root = Path(args.test_root).resolve()
    if not test_root.exists():
        print(f"[FATAL] test root missing: {test_root}")
        return

    summary = { 'classes': {}, 'extremely_off': [] }

    for cls in TARGET_CLASSES:
        cls_dir = test_root / cls
        if not cls_dir.exists():
            print(f"[WARN] Missing class folder: {cls_dir}")
            continue
        wavs = list(cls_dir.rglob('*.wav'))
        wavs.sort()
        total = 0
        correct = 0
        off = 0
        extreme = 0
        details = []
        for w in wavs:
            total += 1
            try:
                audio_bytes = load_audio(w)
                pred = clf.predict(audio_bytes)
                pred_class = pred['class']
                conf = pred['confidence']
                all_probs = pred.get('all_probs', {})
                is_corr = pred_class == cls
                if is_corr:
                    correct += 1
                else:
                    off += 1
                extreme_flag = is_extremely_off(pred_class, conf, cls, all_probs)
                if extreme_flag:
                    extreme += 1
                    summary['extremely_off'].append({
                        'file': str(w),
                        'expected': cls,
                        'pred': pred_class,
                        'confidence': round(conf,4),
                        'expected_prob': round(all_probs.get(cls,0.0),4),
                        'top2': pred['top2']
                    })
                details.append({
                    'file': w.name,
                    'pred': pred_class,
                    'confidence': round(conf,4),
                    'expected_prob': round(all_probs.get(cls,0.0),4),
                    'correct': is_corr,
                    'extremely_off': extreme_flag
                })
            except Exception as e:
                details.append({'file': w.name, 'error': str(e)})
        acc = correct / total if total else 0.0
        summary['classes'][cls] = {
            'total': total,
            'correct': correct,
            'accuracy': round(acc,3),
            'off': off,
            'extremely_off_count': extreme,
            'samples': details
        }
        print(f"Class {cls:16} total={total} acc={acc*100:5.1f}% off={off} extreme={extreme}")

    out_path = Path(args.output)
    with open(out_path,'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved audit to {out_path.resolve()}")
    if summary['extremely_off']:
        print("\nExtremely off examples (up to first 25):")
        for ex in summary['extremely_off'][:25]:
            print(f"  {ex['file']} -> {ex['pred']} (conf={ex['confidence']:.2f}, exp_prob={ex['expected_prob']:.2f})")
    else:
        print("No extremely off samples detected by heuristic.")

if __name__ == '__main__':
    main()
