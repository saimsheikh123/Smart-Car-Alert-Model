import argparse
import json
import os
import shutil

"""
Quarantine extreme-off misclassified test samples and generate label_adjustments.json.

Rules:
  - extreme-off defined in audit JSON (extremely_off == true)
  - Destination: <dataset_root>/noisy_quarantine/<original_class>/<filename>
  - label_adjustments entry per file with fields:
        original_class, predicted_class, confidence, expected_prob, action
  - Action heuristics:
        confidence >= 0.99 and expected_prob <= 0.01 -> consider_relabel_to_pred
        confidence >= 0.95 and expected_prob <= 0.05 -> listen_and_decide
        else -> review
Dry run supported.
"""

def decide_action(confidence: float, expected_prob: float, original: str, predicted: str) -> str:
    if confidence >= 0.99 and expected_prob <= 0.01:
        return "consider_relabel_to_pred"
    if confidence >= 0.95 and expected_prob <= 0.05:
        return "listen_and_decide"
    return "review"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-file", required=True, help="Path to key_class_audit.json")
    parser.add_argument("--test-root", required=True, help="Root of test set (e.g. ..\\train\\dataset\\test)")
    parser.add_argument("--quarantine-root", required=True, help="Destination quarantine root (e.g. ..\\train\\dataset\\noisy_quarantine)")
    parser.add_argument("--label-adjustments", required=True, help="Output JSON with adjustment recommendations")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions, do not move files")
    args = parser.parse_args()

    with open(args.audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)

    classes = audit.get("classes", {})
    adjustments = []
    move_count = 0

    for cls_name, cls_data in classes.items():
        for sample in cls_data.get("samples", []):
            if not sample.get("extremely_off"):
                continue
            filename = sample["file"]
            predicted = sample.get("pred")
            confidence = sample.get("confidence", 0.0)
            expected_prob = sample.get("expected_prob", 0.0)
            original_class = cls_name
            action = decide_action(confidence, expected_prob, original_class, predicted)

            src_path = os.path.join(args.test_root, original_class, filename)
            dst_dir = os.path.join(args.quarantine_root, original_class)
            dst_path = os.path.join(dst_dir, filename)

            adjustments.append({
                "file": filename,
                "original_class": original_class,
                "predicted_class": predicted,
                "confidence": confidence,
                "expected_prob": expected_prob,
                "action": action,
                "source": src_path,
                "quarantine_destination": dst_path
            })

            if not args.dry_run:
                os.makedirs(dst_dir, exist_ok=True)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                    move_count += 1
                else:
                    adjustments[-1]["move_error"] = "source_missing"

    with open(args.label_adjustments, "w", encoding="utf-8") as f:
        json.dump({"adjustments": adjustments}, f, indent=2)

    print(f"Extreme-off samples processed: {len(adjustments)} | Moved: {move_count}")
    print(f"Label adjustments written to {args.label_adjustments}")
    if args.dry_run:
        print("Dry run: no files moved.")


if __name__ == "__main__":
    main()
