"""Compare local ASTClassifier predictions to running server /classify endpoint.

Usage (PowerShell):
  set FORCE_CPU=1
  python compare_server_local.py --model ../ast_6class_model --test-root ../train/dataset/test --server http://127.0.0.1:8010 --limit 10

This helps diagnose discrepancies between `test_6class.py` results and
server responses.
"""
from pathlib import Path
import argparse
import requests
import soundfile as sf
import io
import numpy as np
import librosa
import json
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
        audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='../ast_6class_model')
    ap.add_argument('--test-root', default='../train/dataset/test')
    ap.add_argument('--server', default='http://127.0.0.1:8010')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--per-class-limit', type=int, default=3)
    ap.add_argument('--collision-heuristic', action='store_true')
    args = ap.parse_args()

    if args.collision_heuristic:
        os.environ['COLLISION_HEURISTIC'] = '1'
        print('[INFO] COLLISION_HEURISTIC enabled for local model.')

    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        print(f'[FATAL] Model directory missing: {model_dir}')
        raise SystemExit(2)

    clf = ASTClassifier(str(model_dir), device='cpu')
    test_root = Path(args.test_root).resolve()
    if not test_root.exists():
        print(f'[FATAL] Test root missing: {test_root}')
        raise SystemExit(3)

    files = []
    for cls in CLASSES:
        cls_dir = test_root / cls
        if not cls_dir.exists():
            continue
        wavs = list(cls_dir.rglob('*.wav'))[:args.per_class_limit]
        for w in wavs:
            files.append((cls, w))
    files = files[:args.limit]

    rows = []
    mismatches = 0
    for true_cls, path in files:
        audio_bytes = load_audio_bytes(path)
        local_pred = clf.predict(audio_bytes)

        # Send to server
        try:
            resp = requests.post(f'{args.server}/classify', files={'file': (path.name, audio_bytes, 'audio/wav')}, timeout=15)
            if resp.status_code == 200:
                server_json = resp.json()
                server_pred = server_json['ensemble']['final_prediction']
                server_conf = server_json['ensemble']['confidence']
            else:
                server_pred = f'ERROR_{resp.status_code}'
                server_conf = 0.0
        except Exception as e:
            server_pred = f'EXCEPTION'
            server_conf = 0.0

        local_cls = local_pred['class']
        local_conf = local_pred['confidence']
        match = (local_cls == server_pred)
        if not match:
            mismatches += 1
        rows.append({
            'file': path.name,
            'true': true_cls,
            'local_pred': local_cls,
            'local_conf': round(local_conf, 4),
            'server_pred': server_pred,
            'server_conf': round(server_conf, 4),
            'match': match,
        })
        print(f'{path.name:<30} true={true_cls:<18} local={local_cls:<18} server={server_pred:<18} match={match}')

    summary = {
        'total': len(rows),
        'mismatches': mismatches,
        'rows': rows,
    }
    with open('compare_server_local_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved detailed comparison to compare_server_local_results.json')
    print(f'Mismatches: {mismatches}/{len(rows)}')

if __name__ == '__main__':
    main()
