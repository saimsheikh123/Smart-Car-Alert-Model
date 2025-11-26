"""
Fine-tune Audio Spectrogram Transformer (AST) for 6-class audio classification
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ASTForAudioClassification, ASTFeatureExtractor, TrainingArguments, Trainer
import librosa
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from tqdm import tqdm

# Configuration
CLASS_NAMES = [
    "alert_sounds",
    "collision_sounds",
    "emergency_sirens",
    "environmental_sounds",
    "human_scream",
    "road_traffic",
]

DATASET_PATH = Path(r"C:\Users\Saim\cmpe-281-models\cmpe-281-models\train\dataset")
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUTPUT_DIR = "./ast_model_checkpoints"
FINAL_MODEL_PATH = "./ast_6class_model"

# Hyperparameters (env overridable for quick runs)
def _get_env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def _get_env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

BATCH_SIZE = _get_env_int("BATCH_SIZE", 8)
LEARNING_RATE = _get_env_float("LR", 5e-5)
NUM_EPOCHS = _get_env_int("EPOCHS", 10)
MAX_AUDIO_LENGTH = _get_env_int("MAX_AUDIO_LENGTH", 10)  # seconds
SAMPLING_RATE = _get_env_int("SAMPLING_RATE", 16000)


class AudioDataset(Dataset):
    """Dataset for loading audio files"""
    
    def __init__(self, data_path, class_names, feature_extractor, max_length=10, sampling_rate=16000):
        self.data_path = Path(data_path)
        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        
        # Collect all audio files
        self.samples = []
        for class_name in class_names:
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found, skipping...")
                continue
            
            # Find all audio files
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            for ext in audio_extensions:
                for audio_file in class_dir.glob(f'*{ext}'):
                    self.samples.append({
                        'path': audio_file,
                        'label': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
        
        # Print class distribution
        class_counts = {}
        for sample in self.samples:
            cls = sample['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("\nClass distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count} files")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        try:
            audio, sr = librosa.load(sample['path'], sr=self.sampling_rate, mono=True)
            
            # Pad or truncate to max_length
            max_samples = self.max_length * self.sampling_rate
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
            
            # Extract features using AST feature extractor
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt"
            )
            
            return {
                'input_values': inputs['input_values'].squeeze(0),
                'labels': torch.tensor(sample['label'], dtype=torch.long)
            }
        
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a zero tensor as fallback
            zero_audio = np.zeros(self.max_length * self.sampling_rate)
            inputs = self.feature_extractor(
                zero_audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt"
            )
            return {
                'input_values': inputs['input_values'].squeeze(0),
                'labels': torch.tensor(sample['label'], dtype=torch.long)
            }


def compute_metrics(pred):
    """Compute accuracy metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


def main():
    print("=" * 80)
    print("Audio Spectrogram Transformer (AST) Training")
    print("=" * 80)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load feature extractor
    print(f"\nLoading feature extractor from {MODEL_NAME}...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    
    # Load pretrained model and adapt for 7 classes
    print(f"Loading pretrained model and adapting to {len(CLASS_NAMES)} classes...")
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True
    )
    
    # Freeze early layers (optional - for faster training)
    # Uncomment to freeze all but last 2 transformer layers
    # for name, param in model.named_parameters():
    #     if 'audio_spectrogram_transformer.encoder.layer' in name:
    #         layer_num = int(name.split('.layer.')[1].split('.')[0])
    #         if layer_num < 10:  # Freeze first 10 layers (out of 12)
    #             param.requires_grad = False
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = AudioDataset(
        TRAIN_PATH,
        CLASS_NAMES,
        feature_extractor,
        max_length=MAX_AUDIO_LENGTH,
        sampling_rate=SAMPLING_RATE
    )
    
    test_dataset = AudioDataset(
        TEST_PATH,
        CLASS_NAMES,
        feature_extractor,
        max_length=MAX_AUDIO_LENGTH,
        sampling_rate=SAMPLING_RATE
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        remove_unused_columns=False,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 80)
    
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:")
    print(json.dumps(test_results, indent=2))
    
    # Get detailed predictions for classification report
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids
    
    print("\nDetailed Classification Report:")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=CLASS_NAMES,
        digits=3
    ))
    
    # Save final model
    print(f"\nSaving final model to {FINAL_MODEL_PATH}...")
    model.save_pretrained(FINAL_MODEL_PATH)
    feature_extractor.save_pretrained(FINAL_MODEL_PATH)
    
    # Save class names
    with open(Path(FINAL_MODEL_PATH) / "class_names.json", 'w') as f:
        json.dump(CLASS_NAMES, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Model saved to: {FINAL_MODEL_PATH}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.1%}")
    print("=" * 80)
    
    # Save training summary
    summary = {
        "model_name": MODEL_NAME,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "test_accuracy": test_results['eval_accuracy'],
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "max_audio_length": MAX_AUDIO_LENGTH,
            "sampling_rate": SAMPLING_RATE
        }
    }
    
    with open(Path(FINAL_MODEL_PATH) / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to: {FINAL_MODEL_PATH}/training_summary.json")


if __name__ == "__main__":
    main()
