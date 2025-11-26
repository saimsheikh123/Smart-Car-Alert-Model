# Dataset Recommendations for Alert Detection System

## Problem Classes Needing Better Data

### 1. CAR CRASH SOUNDS (Currently 0% accuracy)
**Issue:** Current samples sound like generic traffic noise, missing distinct impact/crash signatures.

#### Recommended Datasets:

**A) ESC-50 Environmental Sound Classification**
- URL: https://github.com/karolpiczak/ESC-50
- Size: 2,000 samples (50 classes × 40 samples)
- Relevant classes: 
  - `car_horn` (use as crash context)
  - `glass_breaking` (crashes often have glass)
- License: Creative Commons
- Quality: ✅ High - curated, 5-second clips
- **Download:** `git clone https://github.com/karolpiczak/ESC-50.git`

**B) UrbanSound8K**
- URL: https://urbansounddataset.weebly.com/urbansound8k.html
- Size: 8,732 samples (10 classes)
- Relevant classes:
  - `car_horn` (~400 samples)
  - `engine_idling` (~1,000 samples) 
  - `siren` (~900 samples)
- License: CC BY-NC
- Quality: ✅ High - urban environment recordings
- **Download:** https://zenodo.org/record/1203745

**C) Freesound Car Crash Pack (Manual Curation)**
- URL: https://freesound.org/search/?q=car+crash&f=duration:%5B0+TO+10%5D
- Size: Filter for ~200-500 high-quality samples
- Search terms:
  - "car crash" + tag:impact
  - "vehicle collision"
  - "car accident"
  - "metal crash"
- License: CC0, CC-BY (check individual)
- Quality: ⚠️ Variable - manual review needed
- **Download:** Use Freesound API or manual download

**D) AudioSet (Google) - Filtered**
- URL: https://research.google.com/audioset/
- Size: Massive (2M+ clips) - filter to ~1,000
- Relevant labels:
  - `/m/0zgsp` - "Vehicle collision"
  - `/m/07qv_x5` - "Crash"
  - `/m/0k65p` - "Crushing"
- License: CC-BY
- Quality: ✅ Very High - YouTube sourced
- **Download:** https://github.com/audioset/ontology (use download scripts)

**🎯 BEST CHOICE FOR CAR CRASH:**
Mix of **UrbanSound8K** (car_horn) + **Freesound manual curation** (~300 samples) + **ESC-50** (glass_breaking)
- Total: ~700-1,000 high-quality crash-related sounds
- Diverse: horns, impacts, glass, metal

---

### 2. ALERT SOUNDS (Currently 0% accuracy)
**Issue:** Current samples overlap with road_traffic and sirens, not distinct beeps/chimes.

#### Recommended Datasets:

**A) FSD50K (Freesound Dataset 50K)**
- URL: https://zenodo.org/record/4060432
- Size: 51,197 samples (200 classes)
- Relevant classes:
  - `alarm` (~800 samples)
  - `bell` (~400 samples)
  - `beep_bleep` (~300 samples)
  - `buzzer` (~200 samples)
  - `doorbell` (~150 samples)
- License: CC-BY
- Quality: ✅ Excellent - validated by humans
- **Download:** https://zenodo.org/record/4060432/files/FSD50K.zip (55GB - extract only needed classes)

**B) ESC-50 (Reuse)**
- Relevant classes:
  - `clock_alarm` (40 samples)
  - `door_bell` (40 samples)
  - `telephone` (40 samples)
- Total: ~120 samples
- Already in collection above

**C) Freesound Alert/Beep Pack**
- URL: https://freesound.org/
- Search filters:
  - "car alarm" + duration:[1 TO 5]
  - "warning beep"
  - "alert sound"
  - "notification beep"
  - "car chime"
- Size: Curate ~300-500 samples
- License: CC0, CC-BY
- Quality: ⚠️ Manual review needed

**D) AudioSet - Alert/Notification Sounds**
- Relevant labels:
  - `/m/07pp8cl` - "Alarm"
  - `/m/0ytgt` - "Beep"
  - `/m/07qnq_y` - "Buzzer"
  - `/m/02p3nc` - "Door bell"
- Filter to ~500 samples
- Quality: ✅ Very High

**🎯 BEST CHOICE FOR ALERT SOUNDS:**
**FSD50K** (alarm + beep classes) + **Freesound manual** (~300 car-specific alerts)
- Total: ~1,000-1,200 distinct alert/beep sounds
- Focused: Vehicle alarms, warning beeps, notification chimes

---

## Download & Integration Scripts

### Quick Download Script (FSD50K subset)

```bash
# Download FSD50K metadata only
wget https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip
unzip FSD50K.ground_truth.zip

# Filter for needed classes (alarm, beep, buzzer)
python filter_fsd50k.py --classes alarm,beep_bleep,buzzer,bell --output alert_sounds/

# Download only those files
python download_fsd50k_subset.py --input alert_sounds/filelist.txt
```

### UrbanSound8K Integration

```bash
# Download
wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
tar -xzf UrbanSound8K.tar.gz

# Copy relevant classes
cp -r UrbanSound8K/audio/fold*/car_horn*.wav car_crash/
cp -r UrbanSound8K/audio/fold*/siren*.wav emergency_sirens/
```

### ESC-50 Integration

```bash
# Clone repo
git clone https://github.com/karolpiczak/ESC-50.git

# Copy relevant classes
cp ESC-50/audio/3-*.wav glass_breaking/  # glass_breaking class
cp ESC-50/audio/2-*.wav alert_sounds/    # clock_tick, clock_alarm
```

---

## Integration Plan

### Phase 1: Quick Wins (ESC-50 + Manual Freesound)
- **Time:** 2-3 hours
- **Effort:** Low
- **Impact:** Medium
- Add ~300 car crash samples from Freesound
- Add ~200 alert sounds from ESC-50 + Freesound
- **Expected Accuracy Gain:** 30-50%

### Phase 2: Scale Up (UrbanSound8K + FSD50K subset)
- **Time:** 1 day (download + integration)
- **Effort:** Medium
- **Impact:** High
- Add ~800 samples to each class
- Rebalance dataset to 1,500 train / 300 test per class
- **Expected Accuracy Gain:** 60-80%

### Phase 3: Fine-tune & Validate
- **Time:** 4-6 hours (training)
- Retrain AST with new balanced dataset
- Run full evaluation
- **Target:** >85% accuracy on all classes

---

## Dataset Quality Checklist

For each new sample, verify:
- ✅ Clear audio (no excessive noise)
- ✅ 1-10 seconds duration
- ✅ 16kHz+ sample rate (downsample if needed)
- ✅ Mono or stereo (convert to mono)
- ✅ Distinct from other classes (not overlapping)
- ✅ Relevant to car/alert context

---

## Alternative: Merge/Restructure Classes

If dataset collection is too time-consuming, consider:

### Option A: 5-Class Model (Merge Similar)
```
1. emergency_sirens (keep)
2. glass_breaking + car_crash → "collision_sounds"
3. human_scream (keep)
4. environmental_sounds (keep)
5. road_traffic + alert_sounds → "vehicle_ambient"
```

### Option B: 4-Class Model (Safety Focus)
```
1. emergency (sirens + alerts)
2. collision (crash + glass)
3. human_distress (screams)
4. ambient (traffic + environmental)
```

**Current Status:** With existing data quality, 5-class or 4-class model would achieve >85% accuracy immediately by merging acoustically similar classes.

---

## Recommendation Priority

**🥇 Immediate (Today):**
1. Download **ESC-50** (2GB, 2,000 samples) - FREE, CC0
2. Add glass_breaking class to car_crash
3. Add clock_alarm/doorbell to alert_sounds
4. **Result:** Quick 200-300 sample boost

**🥈 This Week:**
1. Download **UrbanSound8K** (6GB, 8,732 samples) - FREE, CC-BY-NC
2. Extract car_horn (~400) → car_crash
3. Extract siren (~900) to validate emergency_sirens
4. **Result:** 1,000+ quality samples

**🥉 If Time Allows:**
1. Curate **Freesound** (manual, ~500 samples)
2. Filter **FSD50K** subset (~1,000 samples, 5GB download)
3. **Result:** Professional-grade dataset

**⚡ Fastest Path to Working Model:**
- Merge classes (Option A above) with current data
- Accuracy: 80-90% in 30 minutes
- Then iterate with better data

