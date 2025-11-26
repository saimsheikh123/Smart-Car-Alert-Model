"""
Analyze confidence levels from full test results
"""
import json
import numpy as np

# Load results
with open('full_test_results.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("CONFIDENCE ANALYSIS BY CLASS")
print("="*80)

classes = ['alert_sounds', 'collision_sounds', 'emergency_sirens', 
           'environmental_sounds', 'human_scream', 'road_traffic']

for cls in classes:
    results = data['per_class_stats'][cls]['results']
    correct = [r for r in results if r['correct']]
    wrong = [r for r in results if not r['correct']]
    
    print(f"\n{cls.upper().replace('_', ' ')}:")
    print(f"  Total files: {len(results)}")
    print(f"  Correct: {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
    
    if correct:
        conf_correct = [r['confidence'] for r in correct]
        print(f"  Confidence (correct predictions):")
        print(f"    Mean: {np.mean(conf_correct):.1%}")
        print(f"    Median: {np.median(conf_correct):.1%}")
        print(f"    Min: {min(conf_correct):.1%}")
        print(f"    Max: {max(conf_correct):.1%}")
        low_conf = [c for c in conf_correct if c < 0.9]
        if low_conf:
            print(f"    <90% confidence: {len(low_conf)} files ({len(low_conf)/len(correct)*100:.1f}%)")
    
    if wrong:
        conf_wrong = [r['confidence'] for r in wrong]
        print(f"  Errors: {len(wrong)} files")
        print(f"  Confidence (wrong predictions):")
        print(f"    Mean: {np.mean(conf_wrong):.1%}")
        print(f"    Median: {np.median(conf_wrong):.1%}")
        print(f"    Range: {min(conf_wrong):.1%} - {max(conf_wrong):.1%}")
        
        # Show error details
        print(f"  Error breakdown:")
        error_map = {}
        for r in wrong:
            key = f"{r['true_class']} → {r['predicted_class']}"
            if key not in error_map:
                error_map[key] = []
            error_map[key].append(r['confidence'])
        
        for error_type, confs in sorted(error_map.items()):
            print(f"    {error_type}: {len(confs)} files (avg conf: {np.mean(confs):.1%})")

print("\n" + "="*80)
print("OVERALL CONFIDENCE SUMMARY")
print("="*80)

all_correct = []
all_wrong = []

for cls_data in data['per_class_stats'].values():
    for r in cls_data['results']:
        if r['correct']:
            all_correct.append(r['confidence'])
        else:
            all_wrong.append(r['confidence'])

print(f"\nCORRECT PREDICTIONS ({len(all_correct)} files, {len(all_correct)/1206*100:.1f}%):")
print(f"  Mean confidence: {np.mean(all_correct):.2%}")
print(f"  Median confidence: {np.median(all_correct):.2%}")
print(f"  Std deviation: {np.std(all_correct):.2%}")
print(f"  Min: {min(all_correct):.2%}")
print(f"  25th percentile: {np.percentile(all_correct, 25):.2%}")
print(f"  50th percentile: {np.percentile(all_correct, 50):.2%}")
print(f"  75th percentile: {np.percentile(all_correct, 75):.2%}")
print(f"  95th percentile: {np.percentile(all_correct, 95):.2%}")

# Confidence distribution
print(f"\n  Confidence distribution:")
bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.0)]
for low, high in bins:
    count = len([c for c in all_correct if low <= c < high])
    print(f"    {low:.0%}-{high:.0%}: {count} files ({count/len(all_correct)*100:.1f}%)")
count_100 = len([c for c in all_correct if c >= 1.0])
print(f"    100%: {count_100} files ({count_100/len(all_correct)*100:.1f}%)")

print(f"\nWRONG PREDICTIONS ({len(all_wrong)} files, {len(all_wrong)/1206*100:.1f}%):")
if all_wrong:
    print(f"  Mean confidence: {np.mean(all_wrong):.2%}")
    print(f"  Median confidence: {np.median(all_wrong):.2%}")
    print(f"  Range: {min(all_wrong):.2%} - {max(all_wrong):.2%}")
    
    # Categorize errors by confidence
    low_conf_errors = [c for c in all_wrong if c < 0.8]
    mid_conf_errors = [c for c in all_wrong if 0.8 <= c < 0.95]
    high_conf_errors = [c for c in all_wrong if c >= 0.95]
    
    print(f"\n  Error confidence breakdown:")
    print(f"    Low confidence (<80%): {len(low_conf_errors)} files ({len(low_conf_errors)/len(all_wrong)*100:.1f}%)")
    print(f"    Medium confidence (80-95%): {len(mid_conf_errors)} files ({len(mid_conf_errors)/len(all_wrong)*100:.1f}%)")
    print(f"    High confidence (≥95%): {len(high_conf_errors)} files ({len(high_conf_errors)/len(all_wrong)*100:.1f}%)")
    
    if high_conf_errors:
        print(f"\n  ⚠️ Model is very confident on {len(high_conf_errors)} wrong predictions - these may need relabeling")

print("\n" + "="*80)
print("CALIBRATION ASSESSMENT")
print("="*80)

# Perfect calibration: confidence matches accuracy
print("\nModel is well-calibrated when:")
print("  - 90% confidence predictions are correct ~90% of the time")
print("  - 99% confidence predictions are correct ~99% of the time")

all_predictions = []
for cls_data in data['per_class_stats'].values():
    all_predictions.extend(cls_data['results'])

conf_bins = [(0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.0)]
print("\nCalibration by confidence bin:")
for low, high in conf_bins:
    in_bin = [p for p in all_predictions if low <= p['confidence'] < high]
    if in_bin:
        correct_in_bin = [p for p in in_bin if p['correct']]
        actual_acc = len(correct_in_bin) / len(in_bin)
        expected_conf = (low + high) / 2
        print(f"  {low:.0%}-{high:.0%} confidence: {len(in_bin)} files, {actual_acc:.1%} actually correct (expected ~{expected_conf:.0%})")

in_bin_100 = [p for p in all_predictions if p['confidence'] >= 1.0]
if in_bin_100:
    correct_100 = [p for p in in_bin_100 if p['correct']]
    print(f"  100% confidence: {len(in_bin_100)} files, {len(correct_100)/len(in_bin_100):.1%} actually correct")

print("\n" + "="*80)
