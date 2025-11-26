#!/usr/bin/env python3
"""
prepare_dataset.py
Organize audio files from multiple sources into train/val/test folders.
Handles format conversion, resampling, and data augmentation for imbalanced classes.

Usage:
    python prepare_dataset.py --config config.yaml
    python prepare_dataset.py --sources ./audio_datasets ./corrections_audio --output ./dataset --val_split 0.15
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil
import yaml

# Augmentation utilities
def add_noise(audio, sr, noise_factor=0.005):
    """Add Gaussian noise to audio."""
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented

def time_stretch(audio, stretch_factor=1.1):
    """Time-stretch audio."""
    return librosa.effects.time_stretch(audio, rate=stretch_factor)

def pitch_shift(audio, sr, n_steps=2):
    """Pitch-shift audio (slower operation)."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def random_shift(audio, shift_max=0.2):
    """Random time-shift within audio."""
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def augment_audio(audio, sr, num_augmentations=2, noise_prob=0.3, stretch_prob=0.3, shift_prob=0.3):
    """Generate augmented versions of audio."""
    augmented_list = [audio]  # Original
    for _ in range(num_augmentations):
        aug = audio.copy()
        if np.random.rand() < noise_prob:
            aug = add_noise(aug, sr, noise_factor=0.005)
        if np.random.rand() < stretch_prob:
            stretch_factor = np.random.uniform(0.8, 1.2)
            aug = time_stretch(aug, stretch_factor)
            if len(aug) != len(audio):
                aug = np.pad(aug, (0, max(0, len(audio) - len(aug))), mode='constant')[:len(audio)]
        if np.random.rand() < shift_prob:
            aug = random_shift(aug, shift_max=0.1)
        augmented_list.append(aug)
    return augmented_list

def resample_and_normalize(audio_path, sr=16000, duration=3.0):
    """Load, resample, and normalize audio to fixed duration."""
    try:
        y, file_sr = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
        # Pad or trim to exact duration
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
        return y
    except Exception as e:
        print(f"  [ERROR] Failed to load {audio_path}: {e}")
        return None

def organize_dataset(sources, output_dir, config, class_mapping=None, augment_minority=True):
    """
    Organize audio files from multiple sources into class folders.
    
    Args:
        sources (list): Paths to source directories or metadata files
        output_dir (str): Root output directory (will create train/val/test subfolders)
        config (dict): Configuration dict with dataset settings
        class_mapping (dict): Optional mapping of source filenames to class labels
        augment_minority (bool): Whether to augment underrepresented classes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    classes = config['classes']
    sr = config['sample_rate']
    duration = config['duration']
    train_split = config.get('train_split', 0.70)
    val_split = config.get('val_split', 0.15)
    
    # Create class subdirectories
    for split in ['train', 'val', 'test']:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Index: class -> list of file paths
    class_files = defaultdict(list)
    
    print("[1/3] Scanning source directories for audio files...")
    for source in sources:
        source_path = Path(source)
        if not source_path.exists():
            print(f"  [WARNING] Source not found: {source}")
            continue
        
        print(f"  Scanning: {source}")
        
        # If source is a JSON metadata file (from feedback), parse it
        if source_path.suffix == '.json':
            try:
                with open(source_path, 'r') as f:
                    corrections = json.load(f)
                for item in corrections:
                    audio_file = item.get('audio_path') or item.get('audio_filename')
                    actual_class = item.get('actual')
                    if audio_file and actual_class and actual_class in classes:
                        if Path(audio_file).exists():
                            class_files[actual_class].append(audio_file)
            except Exception as e:
                print(f"  [ERROR] Failed to parse JSON {source}: {e}")
        
        # Otherwise, scan directory recursively for all audio files
        elif source_path.is_dir():
            # First, try to match top-level subdirectories to class names
            for item in source_path.iterdir():
                if item.is_dir():
                    folder_name_lower = item.name.lower().replace('_', ' ').replace('-', ' ')
                    # Match folder name to class name
                    matched_class = None
                    for cls in classes:
                        cls_normalized = cls.lower().replace('_', ' ')
                        if cls_normalized in folder_name_lower or folder_name_lower in cls_normalized:
                            matched_class = cls
                            break
                    
                    if matched_class:
                        # Recursively find all audio files in this directory
                        for audio_file in item.glob('**/*.wav'):
                            class_files[matched_class].append(str(audio_file))
                        for audio_file in item.glob('**/*.mp3'):
                            class_files[matched_class].append(str(audio_file))
            
            # Also recursively search all audio files and infer class from path/filename
            for audio_file in source_path.glob('**/*.wav'):
                filename_lower = audio_file.stem.lower()
                for cls in classes:
                    if cls.replace('_', ' ') in filename_lower or cls in filename_lower:
                        if str(audio_file) not in class_files[cls]:  # Avoid duplicates
                            class_files[cls].append(str(audio_file))
                        break
    
    # Print summary
    print("\n[2/3] Dataset summary:")
    total_files = sum(len(v) for v in class_files.values())
    print(f"  Total audio files found: {total_files}")
    for cls in classes:
        count = len(class_files[cls])
        print(f"    {cls}: {count} files")
    
    # Compute class weights for augmentation
    if augment_minority:
        max_count = max(len(v) for v in class_files.values()) if class_files else 1
        class_augment_counts = {}
        for cls in classes:
            count = len(class_files[cls])
            # Augment minority classes to ~80% of max
            target = int(max_count * 0.8)
            if count > 0 and count < target:
                class_augment_counts[cls] = target - count
            else:
                class_augment_counts[cls] = 0
        print(f"\n  Augmentation targets (to balance classes):")
        for cls, aug_count in class_augment_counts.items():
            if aug_count > 0:
                print(f"    {cls}: +{aug_count} augmented files")
    
    # Process and split files
    print("\n[3/3] Converting, augmenting, and splitting audio...")
    split_counts = defaultdict(lambda: defaultdict(int))
    
    for cls in classes:
        files = class_files[cls]
        if not files:
            print(f"  [WARNING] No files found for class: {cls}")
            continue
        
        print(f"  Processing {cls}...")
        
        # Determine split indices
        n_files = len(files)
        n_train = int(n_files * train_split)
        n_val = int(n_files * val_split)
        
        # Shuffle
        np.random.shuffle(files)
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Process each split
        for split, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            output_class_dir = output_path / split / cls
            
            for file_idx, audio_file in enumerate(tqdm(split_files, desc=f"  {split}/{cls}", leave=False)):
                try:
                    audio = resample_and_normalize(audio_file, sr=sr, duration=duration)
                    if audio is None:
                        continue
                    
                    # Save original
                    base_name = Path(audio_file).stem
                    output_file = output_class_dir / f"{base_name}.wav"
                    sf.write(output_file, audio, sr)
                    split_counts[split][cls] += 1
                    
                    # Augment training data for minority classes
                    if split == 'train' and augment_minority and augment_minority and cls in class_augment_counts:
                        if class_augment_counts[cls] > 0:
                            num_aug = max(1, class_augment_counts[cls] // len(train_files))
                            augmented_audios = augment_audio(audio, sr, num_augmentations=num_aug)
                            for aug_idx, aug_audio in enumerate(augmented_audios[1:], start=1):
                                aug_file = output_class_dir / f"{base_name}_aug{aug_idx}.wav"
                                sf.write(aug_file, aug_audio, sr)
                                split_counts[split][cls] += 1
                
                except Exception as e:
                    print(f"    [ERROR] Failed to process {audio_file}: {e}")
    
    # Print final summary
    print("\nDataset organization complete!")
    print("Final split counts:")
    for split in ['train', 'val', 'test']:
        print(f"  {split}:")
        for cls in classes:
            count = split_counts[split][cls]
            print(f"    {cls}: {count}")
    
    # Save manifest
    manifest = {
        'classes': classes,
        'root_dir': str(output_path),
        'splits': dict(split_counts),
        'sample_rate': sr,
        'duration': duration
    }
    manifest_file = output_path / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_file}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file (YAML)')
    parser.add_argument('--sources', type=str, nargs='+', help='Source directories (overrides config)')
    parser.add_argument('--output', type=str, help='Output directory (overrides config)')
    parser.add_argument('--train_split', type=float, default=0.70, help='Train split ratio')
    parser.add_argument('--val_split', type=float, default=0.15, help='Val split ratio')
    parser.add_argument('--augment_minority', action='store_true', default=True, help='Augment underrepresented classes')
    args = parser.parse_args()
    
    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get('dataset', {})
    
    # Override with CLI args
    if args.sources:
        sources = args.sources
    else:
        sources = [
            './audio_datasets',
            './corrections_audio'
        ]
    
    output_dir = args.output or config.get('root_dir', './dataset')
    config['train_split'] = args.train_split
    config['val_split'] = args.val_split
    
    # Set defaults
    config.setdefault('sample_rate', 16000)
    config.setdefault('duration', 3.0)
    config.setdefault('classes', ['alert_sounds', 'car_crash', 'emergency_sirens', 'environmental_sounds', 'glass_breaking', 'human_scream', 'road_traffic'])
    
    print(f"[INFO] Using config: {args.config}")
    print(f"[INFO] Source directories: {sources}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Train/Val/Test split: {args.train_split}/{args.val_split}/{1-args.train_split-args.val_split}")
    
    organize_dataset(sources, output_dir, config, augment_minority=args.augment_minority)

if __name__ == '__main__':
    main()
