"""
Integration module for Audio Spectrogram Transformer (AST)
Drop-in replacement for AudioCRNN in multi_model_api.py
"""

import torch
import librosa
import numpy as np
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from pathlib import Path
import json
import os

class ASTClassifier:
    """Audio Spectrogram Transformer Classifier"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = Path(model_path)
        
        print(f"[AST] Loading model from {model_path}...")
        
        # Load model and feature extractor
        self.model = ASTForAudioClassification.from_pretrained(model_path)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Load class names
        class_names_file = self.model_path / "class_names.json"
        if class_names_file.exists():
            with open(class_names_file, 'r') as f:
                self.class_names = json.load(f)
        else:
            # Fallback to default 6 classes
            self.class_names = [
                "alert_sounds",
                "collision_sounds",
                "emergency_sirens",
                "environmental_sounds",
                "human_scream",
                "road_traffic",
            ]

        # Pretrained labels (for zero-shot-style mapping when not fine-tuned)
        # If the loaded model has many labels (e.g., 527 from AudioSet), we map them to our 7 classes
        self.pretrained_id2label = getattr(self.model.config, "id2label", None)
        if isinstance(self.pretrained_id2label, dict):
            # Normalize to list of strings
            self.pretrained_labels = [self.pretrained_id2label[i] for i in range(len(self.pretrained_id2label))]
        else:
            self.pretrained_labels = []

        # Build simple keyword heuristics to aggregate pretrained labels into our 7 classes
        self.keyword_mapping = {
            "emergency_sirens": ["siren", "police", "ambulance", "fire truck", "emergency vehicle"],
            "collision_sounds": ["glass", "shatter", "breaking glass", "car crash", "skid", "collision", "wreck", "crash", "impact"],
            "human_scream": ["scream", "screaming", "shriek", "yell"],
            "alert_sounds": ["alarm", "buzzer", "beep", "beeping", "alert", "chime", "notification"],
            "road_traffic": ["engine", "car", "vehicle", "traffic", "road", "bus", "truck", "horn"],
            "environmental_sounds": ["rain", "wind", "storm", "thunder", "water", "birds", "insect"],
        }
        
        print(f"[AST] Model loaded successfully. Classes: {self.class_names}")
    
    def predict(self, audio_bytes, max_length=10, sampling_rate=16000):
        """
        Predict class from audio bytes
        
        Args:
            audio_bytes: Raw audio data
            max_length: Maximum audio length in seconds
            sampling_rate: Target sampling rate
        
        Returns:
            dict: Prediction results
        """
        try:
            # Load audio from bytes
            import soundfile as sf
            import io
            
            audio_io = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_io)
            
            # Resample if needed
            if sr != sampling_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Pad or truncate
            max_samples = max_length * sampling_rate
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
            
            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Two modes:
            # 1) Fine-tuned local model with 7 labels -> direct mapping
            # 2) Pretrained AudioSet model with many labels -> aggregate via keyword mapping
            if len(probs) == len(self.class_names):
                # Direct 7-class prediction (fine-tuned model)
                idx = int(np.argmax(probs))
                predicted_class = self.class_names[idx]
                confidence = float(probs[idx])
                order = np.argsort(probs)[::-1]
                top2 = [
                    {"class": self.class_names[int(order[0])], "confidence": float(probs[int(order[0])])},
                    {"class": self.class_names[int(order[1])], "confidence": float(probs[int(order[1])])},
                ]
                all_probs = {cls: float(prob) for cls, prob in zip(self.class_names, probs)}

                # Optional collision override heuristic
                if os.environ.get("COLLISION_HEURISTIC", "0") == "1":
                    if predicted_class in ("road_traffic", "environmental_sounds"):
                        second_idx = int(order[1])
                        second_class = self.class_names[second_idx]
                        second_conf = float(probs[second_idx])
                        top_conf = float(probs[int(order[0])])
                        # If collision_sounds is close runner-up, promote it
                        if second_class == "collision_sounds" and (top_conf - second_conf) <= 0.08 and second_conf >= 0.15:
                            predicted_class = "collision_sounds"
                            confidence = second_conf
                            top2[0] = {"class": predicted_class, "confidence": confidence}
                            # Keep original top as second for transparency
                            top2[1] = {"class": self.class_names[int(order[0])], "confidence": top_conf}
                            all_probs["_collision_override"] = True
            else:
                # Aggregate pretrained labels into our 7 classes
                label_strings = self.pretrained_labels or [str(i) for i in range(len(probs))]
                agg_scores = {cls: 0.0 for cls in self.class_names}
                for i, p in enumerate(probs):
                    label = label_strings[i].lower()
                    for target, keywords in self.keyword_mapping.items():
                        if any(kw in label for kw in keywords):
                            agg_scores[target] += float(p)
                # Normalize if all zeros (no keyword matches)
                if all(v == 0.0 for v in agg_scores.values()):
                    # Fallback: treat as environmental/road by default
                    agg_scores["environmental_sounds"] = float(np.max(probs))
                # Choose best class
                predicted_class = max(agg_scores, key=agg_scores.get)
                confidence = float(agg_scores[predicted_class])
                # Build top2
                sorted_pairs = sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)
                top2 = [
                    {"class": sorted_pairs[0][0], "confidence": float(sorted_pairs[0][1])},
                    {"class": sorted_pairs[1][0], "confidence": float(sorted_pairs[1][1])},
                ]
                all_probs = agg_scores

                # Optional heuristic also for aggregation mode
                if os.environ.get("COLLISION_HEURISTIC", "0") == "1":
                    if predicted_class in ("road_traffic", "environmental_sounds") and len(sorted_pairs) > 1:
                        second_class, second_conf = sorted_pairs[1][0], float(sorted_pairs[1][1])
                        top_conf = float(sorted_pairs[0][1])
                        if second_class == "collision_sounds" and (top_conf - second_conf) <= 0.08 and second_conf >= 0.15:
                            predicted_class = "collision_sounds"
                            confidence = second_conf
                            top2[0] = {"class": predicted_class, "confidence": confidence}
                            top2[1] = {"class": sorted_pairs[0][0], "confidence": top_conf}
                            all_probs["_collision_override"] = True
            
            # Debug log
            prob_str = ", ".join([f"{c}: {p:.1%}" for c, p in all_probs.items()])
            print(f"[AST] Predicted: {predicted_class} ({confidence:.1%}) | All (6-class agg): {prob_str}")
            
            return {
                "model": "AudioSpectrogramTransformer",
                "class": predicted_class,
                "confidence": confidence,
                "all_probs": all_probs,
                "top2": top2,
                "success": True
            }
        
        except Exception as e:
            print(f"[ERROR] AST prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "model": "AudioSpectrogramTransformer",
                "class": "unknown",
                "confidence": 0.0,
                "all_probs": {},
                "top2": [],
                "success": False,
                "error": str(e)
            }


# For backward compatibility with existing API code
def load_ast_model(model_path, device='cpu'):
    """Load AST model (compatible with existing API structure)"""
    return ASTClassifier(model_path, device)


def predict_ast(model, audio_bytes):
    """Predict using AST model (compatible with existing API structure)"""
    return model.predict(audio_bytes)
