# Emergency Sirens Focused Fine-Tune Plan

## Objective
Raise emergency_sirens test accuracy (>95%) by cleaning mislabeled/noisy samples and emphasizing modulation diversity (wail, yelp, hi–lo, short bursts).

## Data Actions
1. Quarantine 17 extreme-off test samples (completed) — excluded from evaluation and potential relabel review.
2. Listen & tag each quarantined siren-related file: classify into (true_siren | non_siren_alert | ambient | collision artifact | scream-like).
3. Relabel rules:
   - true_siren mislabeled as alert/environmental/collision -> relabel to emergency_sirens.
   - non_siren beep mistakenly in sirens -> move to alert_sounds.
   - broadband impact in sirens -> move to collision_sounds.
   - ambiguous tonal drone lacking modulation -> exclude from training (keep in quarantine).
4. Augment siren set:
   - Time-stretch factors: 0.95x, 1.05x (preserve modulation feel).
   - Pitch shift: -1, +1 semitone.
   - Random 300–800 ms gain dips (simulate occlusion) on 30% of augmented copies.
   - Light band-stop filter (Q≈4, remove narrow 1–2 kHz band) on 15% to simulate distance.

## Batch Construction
Balanced mini-batches (target size 32) approximate distribution:
```
 emergency_sirens: 10
 collision_sounds: 5
 alert_sounds: 5
 road_traffic: 4
 human_scream: 4
 environmental_sounds: 4
```
Oversample emergency_sirens until its effective count ~ other top classes.

## Training Hyperparameters
| Param              | Value            | Rationale                               |
|--------------------|------------------|------------------------------------------|
| EPOCHS             | 4                | Short focused fine-tune                  |
| LR (AdamW)         | 3e-5             | Gentle adaptation                        |
| WEIGHT DECAY       | 0.01             | Regularization                           |
| WARMUP STEPS       | 50               | Stabilize early updates                  |
| MAX_AUDIO_LENGTH   | 10s              | Preserve full siren cycles               |
| MIXUP              | Disabled         | Avoid smearing modulation patterns       |
| DROPOUT (head)     | 0.1              | Slight regularization                    |

## Curriculum
Epoch 1: Full clean set + new augmentations.
Epoch 2: Inject 50% hard siren misclassified (pre-corrected) examples twice.
Epoch 3: Add occlusion & band-stop augmented sirens.
Epoch 4: Reduced LR (1e-5) final convergence.

## Evaluation Checklist
1. Run `test_6class.py` (heuristic off) — target emergency_sirens ≥95%.
2. Re-run `audit_key_classes.py` — confirm reduction in extreme-off sirens.
3. Spot check 5 newly relabeled files using spectrogram visualization.
4. Ensure no class accuracy regresses >2% relative to prior baseline.

## Rollback Strategy
Maintain previous fine-tuned checkpoint; only replace production if sirens ≥95% and other classes remain ≥93%.

## Next Optional Enhancements
1. Contrastive embedding loss (siren vs alert beeps) for tighter boundary.
2. Add energy modulation feature (rate of amplitude envelope change) as auxiliary head.
3. Deploy active learning loop: capture low confidence siren predictions from live traffic and re-label.
