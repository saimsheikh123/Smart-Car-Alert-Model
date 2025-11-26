"""Comprehensive test of ALL files in test dataset (not sampled).

Reports per-class accuracy, confusion matrix, and detailed misclassifications
to determine if retraining is needed.

Usage:
  cd Audio_Models
  set FORCE_CPU=1
  python test_full_dataset.py --model ../ast_6class_model --test-root ../train/dataset/test
"""

from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
import io
import json
import librosa
import os
from collections import defaultdict, Counter
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
    """Load and preprocess audio file to bytes."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    max_samples = max_len_s * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def calculate_metrics(true_labels, pred_labels, classes):
    """Calculate precision, recall, F1 per class."""
    metrics = {}
    for cls in classes:
        true_pos = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        false_pos = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        false_neg = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cls] = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'true_positive': true_pos,
            'false_positive': false_pos,
            'false_negative': false_neg
        }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='../ast_6class_model')
    ap.add_argument('--test-root', default='../train/dataset/test')
    ap.add_argument('--collision-heuristic', action='store_true')
    ap.add_argument('--output', default='full_test_results.json')
    args = ap.parse_args()

    if args.collision_heuristic:
        os.environ['COLLISION_HEURISTIC'] = '1'
        print('[INFO] COLLISION_HEURISTIC enabled.')

    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        print(f"[FATAL] Model directory not found: {model_dir}")
        raise SystemExit(2)

    clf = ASTClassifier(str(model_dir), device='cpu')

    test_root = Path(args.test_root).resolve()
    if not test_root.exists():
        print(f"[FATAL] Test root does not exist: {test_root}")
        raise SystemExit(3)

    print("=" * 80)
    print("COMPREHENSIVE TEST - ALL FILES IN TEST SET")
    print("=" * 80)
    print(f"Model: {model_dir}")
    print(f"Test data: {test_root}")
    print(f"Collision heuristic: {'ON' if args.collision_heuristic else 'OFF'}")
    print("=" * 80)

    all_results = []
    per_class_stats = {}
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    all_true_labels = []
    all_pred_labels = []
    
    total_files = 0
    total_correct = 0
    
    # Process each class
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = test_root / cls
        if not cls_dir.exists():
            print(f"[WARN] Class directory not found: {cls_dir}")
            continue
        
        wavs = list(cls_dir.rglob('*.wav'))
        if not wavs:
            print(f"[WARN] No WAV files found in {cls}")
            continue
        
        print(f"\n[{cls_idx+1}/{len(CLASSES)}] Testing {cls} ({len(wavs)} files)...")
        
        correct = 0
        class_results = []
        
        for idx, wav_path in enumerate(wavs, 1):
            try:
                audio_bytes = load_audio_bytes(wav_path)
                pred = clf.predict(audio_bytes)
                pred_class = pred['class']
                conf = pred['confidence']
                
                is_correct = (pred_class == cls)
                if is_correct:
                    correct += 1
                    total_correct += 1
                
                total_files += 1
                all_true_labels.append(cls)
                all_pred_labels.append(pred_class)
                confusion_matrix[cls][pred_class] += 1
                
                result = {
                    'file': wav_path.name,
                    'true_class': cls,
                    'predicted_class': pred_class,
                    'confidence': round(conf, 4),
                    'correct': is_correct,
                    'top2': pred.get('top2', [])
                }
                class_results.append(result)
                
                # Progress indicator every 50 files
                if idx % 50 == 0:
                    print(f"  Processed {idx}/{len(wavs)} files...")
                    
            except Exception as e:
                print(f"  ERROR processing {wav_path.name}: {e}")
                result = {
                    'file': wav_path.name,
                    'true_class': cls,
                    'error': str(e),
                    'correct': False
                }
                class_results.append(result)
        
        accuracy = correct / len(wavs) if wavs else 0.0
        per_class_stats[cls] = {
            'total_files': len(wavs),
            'correct': correct,
            'accuracy': round(accuracy, 4),
            'results': class_results
        }
        
        print(f"  {cls}: {correct}/{len(wavs)} correct ({accuracy*100:.2f}%)")
        
        # Show top misclassifications for this class
        mistakes = [r for r in class_results if not r['correct'] and 'error' not in r]
        if mistakes:
            mistake_counter = Counter(r['predicted_class'] for r in mistakes)
            print(f"  Most common mistakes:")
            for wrong_class, count in mistake_counter.most_common(3):
                print(f"    {cls} -> {wrong_class}: {count} times ({count/len(wavs)*100:.1f}%)")
    
    # Calculate overall metrics
    overall_accuracy = total_correct / total_files if total_files > 0 else 0.0
    metrics = calculate_metrics(all_true_labels, all_pred_labels, CLASSES)
    
    # Calculate macro and weighted averages
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    weighted_f1 = sum(m['f1'] * per_class_stats[cls]['total_files'] 
                      for cls, m in metrics.items()) / total_files if total_files > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Total files tested: {total_files}")
    print(f"Overall accuracy: {overall_accuracy*100:.2f}% ({total_correct}/{total_files})")
    print(f"Macro-average F1: {macro_f1:.3f}")
    print(f"Weighted-average F1: {weighted_f1:.3f}")
    
    print("\n" + "=" * 80)
    print("PER-CLASS METRICS")
    print("=" * 80)
    print(f"{'Class':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)
    for cls in CLASSES:
        if cls in per_class_stats and cls in metrics:
            stats = per_class_stats[cls]
            m = metrics[cls]
            print(f"{cls:<22} {stats['accuracy']:<10.3f} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1']:<10.3f}")
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 80)
    header = "True/Predicted".ljust(22) + "".join(f"{c[:8]:<10}" for c in CLASSES)
    print(header)
    print("-" * 80)
    for true_cls in CLASSES:
        row = f"{true_cls:<22}"
        for pred_cls in CLASSES:
            count = confusion_matrix[true_cls][pred_cls]
            row += f"{count:<10}"
        print(row)
    
    # Identify problem areas
    print("\n" + "=" * 80)
    print("PROBLEM ANALYSIS & RETRAIN RECOMMENDATION")
    print("=" * 80)
    
    problems = []
    retrain_needed = False
    
    for cls in CLASSES:
        if cls not in per_class_stats:
            continue
        stats = per_class_stats[cls]
        m = metrics[cls]
        
        # Check for poor performance
        if stats['accuracy'] < 0.80:
            problems.append(f"⚠️  {cls}: Low accuracy ({stats['accuracy']*100:.1f}%)")
            retrain_needed = True
        
        if m['f1'] < 0.75:
            problems.append(f"⚠️  {cls}: Low F1 score ({m['f1']:.3f})")
            retrain_needed = True
        
        # Check for specific confusion patterns
        mistakes = [r for r in stats['results'] if not r['correct'] and 'error' not in r]
        if mistakes:
            mistake_counter = Counter(r['predicted_class'] for r in mistakes)
            for wrong_class, count in mistake_counter.most_common(1):
                confusion_rate = count / stats['total_files']
                if confusion_rate > 0.15:  # More than 15% confused with one class
                    problems.append(f"⚠️  {cls} frequently confused with {wrong_class} ({confusion_rate*100:.1f}%)")
                    retrain_needed = True
    
    if problems:
        print("Issues found:")
        for p in problems:
            print(f"  {p}")
    else:
        print("✅ Model performance is good across all classes!")
    
    print("\n" + "-" * 80)
    if retrain_needed:
        print("🔴 RECOMMENDATION: RETRAINING NEEDED")
        print("\nReasons:")
        print("  - One or more classes have accuracy < 80%")
        print("  - One or more classes have F1 score < 0.75")
        print("  - High confusion between specific class pairs")
        print("\nNext steps:")
        print("  1. Review misclassified samples in full_test_results.json")
        print("  2. Check for labeling errors in confused samples")
        print("  3. Consider data augmentation for weak classes")
        print("  4. Retrain with focused curriculum on problem classes")
    else:
        print("🟢 RECOMMENDATION: RETRAINING NOT REQUIRED")
        print("\nModel is performing well. Optional improvements:")
        print("  - Fine-tune only if targeting specific edge cases")
        print("  - Deploy current model for production testing")
    
    # Save detailed results
    output = {
        'test_config': {
            'model_path': str(model_dir),
            'test_root': str(test_root),
            'collision_heuristic': args.collision_heuristic,
            'total_files': total_files
        },
        'overall': {
            'accuracy': round(overall_accuracy, 4),
            'total_correct': total_correct,
            'total_files': total_files,
            'macro_f1': round(macro_f1, 4),
            'weighted_f1': round(weighted_f1, 4)
        },
        'per_class_stats': per_class_stats,
        'per_class_metrics': metrics,
        'confusion_matrix': {k: dict(v) for k, v in confusion_matrix.items()},
        'problems': problems,
        'retrain_needed': retrain_needed
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Detailed results saved to: {output_path.resolve()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
